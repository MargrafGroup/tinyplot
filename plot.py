import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, approx_fprime
import base64
import io

from pyscript import web, when, display, window

"""
    HTML elements:
    #btn-plot
    #btn-fit
    #btn-export

"""

current_fig = None


def get_plot_type():
    # Select the checked radio input
    checked = web.page.find('input[name="plot-type"]:checked')
    if len(checked) > 0:
        return checked[0].value
    return None


def get_loss_type():
    # Select the checked radio input
    checked = web.page.find('input[name="loss-type"]:checked')
    if len(checked) > 0:
        return checked[0].value
    return "ols"


def calculate_metrics(y, y_pred, err, loss_type):
    """
    Calculate R² and -log(likelihood) metrics.
    
    For OLS: unweighted R²
    For WLS: weighted R²
    """
    if loss_type == "wls" and err is not None:
        # Weighted R² for WLS
        # Guard against zero/near-zero errors
        err_safe = np.maximum(err, 1e-10)
        weights = 1.0 / err_safe**2
        y_mean_weighted = np.sum(weights * y) / np.sum(weights)
        ss_res = np.sum(weights * (y - y_pred)**2)
        ss_tot = np.sum(weights * (y - y_mean_weighted)**2)
    else:
        # Unweighted R² for OLS
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
    
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
    
    # Calculate -log(likelihood) based on residuals
    n = len(y)
    sigma2 = ss_res / (n - 2) if n > 2 else ss_res / n
    negloglik = n/2 * np.log(2 * np.pi * sigma2) + ss_res / (2 * sigma2) if sigma2 > 0 else 0
    
    return r2, negloglik


def get_data(plottype):
    data = np.array([list(r) for r in window.csv_rows][1:], dtype=np.float64).T
    if plottype == "scatter":
        x_index = int(web.page["x-select"].value)
        y_index = int(web.page["y-select"].value)
        x = data[x_index, :]
        y = data[y_index, :]
        err_select = web.page["err-select"].value
        err = None
        if err_select and err_select != "" and err_select != "0":
            err_index = int(err_select) - 1
            err = data[err_index, :]
        return x, y, err
    elif plottype == "hist":
        x_index = int(web.page["hist-column"].value)
        return data[x_index, :]


def get_function(plottype="scatter"):
    if plottype == "scatter":
        func_type = web.page["fit-func"].value
        if func_type == "none":
            return None, (0.0)
        elif func_type == "linear":
            return polynomial, (
                float(web.page["par-linear-b"].value),
                float(web.page["par-linear-m"].value),
            )
        elif func_type == "polynomial":
            degree = int(web.page["poly-degree"].value)
            pars = [float(web.page[f"par-poly-a{i}"].value) for i in range(degree + 1)]
            return polynomial, pars
        elif func_type == "exponential":
            return exponential, (
                float(web.page["par-exp-A"].value),
                float(web.page["par-exp-k"].value),
                float(web.page["par-exp-b"].value),
            )
        elif func_type == "logarithmic":
            return logarithmic, (
                float(web.page["par-log-A"].value),
                float(web.page["par-log-b"].value),
            )
    elif plottype == "hist":
        func_type = web.page["hist-fit"].value
        if func_type == "none":
            return None, (0.0)
        elif func_type == "gaussian":
            return gaussian, (
                float(web.page["par-gauss-mu"].value),
                float(web.page["par-gauss-sig"].value),
            )
        elif func_type == "lognormal":
            return lognormal, (
                float(web.page["par-lognorm-mu"].value),
                float(web.page["par-lognorm-sig"].value),
            )
        elif func_type == "mb-2d":
            mb2d = MaxwellBoltzmann(dim=2)
            return mb2d, [float(web.page["par-mb2-a"].value)]
        elif func_type == "mb-3d":
            mb3d = MaxwellBoltzmann(dim=3)
            return mb3d, [float(web.page["par-mb3-a"].value)]


def update_function(parameters, uncertainties, message, plottype="scatter"):
    if plottype == "scatter":
        func_type = web.page["fit-func"].value
        if func_type == "none":
            return None
        elif func_type == "linear":
            web.page["par-linear-b"].value = str(parameters[0])
            web.page["par-linear-m"].value = str(parameters[1])
        elif func_type == "polynomial":
            for i, pi in enumerate(parameters):
                web.page[f"par-poly-a{i}"].value = str(pi)
        elif func_type == "exponential":
            web.page["par-exp-A"].value = str(parameters[0])
            web.page["par-exp-k"].value = str(parameters[1])
            web.page["par-exp-b"].value = str(parameters[2])
        elif func_type == "logarithmic":
            web.page["par-log-A"].value = str(parameters[0])
            web.page["par-log-b"].value = str(parameters[1])
    elif plottype == "hist":
        func_type = web.page["hist-fit"].value
        if func_type == "none":
            return None
        elif func_type == "gaussian":
            web.page["par-gauss-mu"].value = str(parameters[0])
            web.page["par-gauss-sig"].value = str(parameters[1])
        elif func_type == "lognormal":
            web.page["par-lognorm-mu"].value = str(parameters[0])
            web.page["par-lognorm-sig"].value = str(parameters[1])
        elif func_type == "mb-2d":
            web.page["par-mb2-a"].value = str(parameters[0])
        elif func_type == "mb-3d":
            web.page["par-mb3-a"].value = str(parameters[0])

    # Write fit parameters + uncertainties into the result summary widget
    model_name = ""
    equation = ""
    param_names = []

    if plottype == "scatter":
        func_type = web.page["fit-func"].value
        if func_type == "linear":
            model_name = "Linear"
            equation = r"$$f(x) = m x + b$$"
            param_names = ["b", "m"]
        elif func_type == "polynomial":
            degree = int(web.page["poly-degree"].value)
            model_name = f"Polynomial (degree {degree})"
            terms = [f"a_{{{i}}} x^{i}" if i > 0 else "a_{0}" for i in range(degree + 1)]
            equation = r"$$f(x) = " + " + ".join(terms) + "$$"
            param_names = [f"a{i}" for i in range(degree + 1)]
        elif func_type == "exponential":
            model_name = "Exponential"
            equation = r"$$f(x) = A e^{k x} + b$$"
            param_names = ["A", "k", "b"]
        elif func_type == "logarithmic":
            model_name = "Logarithmic"
            equation = r"$$f(x) = A \log(x) + b$$"
            param_names = ["A", "b"]
        else:
            model_name = "None"
            equation = "-"
            param_names = []

    elif plottype == "hist":
        func_type = web.page["hist-fit"].value
        if func_type == "gaussian":
            model_name = "Gaussian"
            equation = r"$$p(x) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$"
            param_names = ["μ", "σ"]
        elif func_type == "lognormal":
            model_name = "Log-normal"
            equation = r"$$p(x) = \frac{1}{x\sigma\sqrt{2\pi}} \exp\left(-\frac{(\ln x-\mu)^2}{2\sigma^2}\right)$$"
            param_names = ["μ", "σ"]
        elif func_type == "mb-2d":
            model_name = "Maxwell–Boltzmann 2D"
            # p = v / a * np.exp(-(v**2) / (2 * a))
            equation = r"$$p(x) = \frac{x}{a} \exp\left(-\frac{x^2}{2a}\right)$$"
            param_names = ["a"]
        elif func_type == "mb-3d":
            model_name = "Maxwell–Boltzmann 3D"
            #p = np.sqrt(2.0 / np.pi) * v**2 / a**3 * np.exp(-(v**2) / (2 * a**2))
            equation = r"$$p(x) = \sqrt{\frac{2}{\pi}} \frac{x^2}{a^3} \exp\left(-\frac{x^2}{2a^2}\right)$$"
            param_names = ["a"]
        else:
            model_name = "None"
            equation = "-"
            param_names = []

    web.page["res-model"].textContent = model_name
    web.page["res-equation"].textContent = equation
    
    # Trigger MathJax to render the LaTeX equation
    try:
        window.MathJax.typeset([window.document.getElementById("res-equation")])
    except Exception:
        pass

    if parameters is not None and len(parameters) > 0 and len(param_names) == len(parameters):
        formatted = []
        for i, (name, p) in enumerate(zip(param_names, parameters)):
            err = uncertainties[i] if uncertainties is not None and i < len(uncertainties) else None
            if err is not None:
                formatted.append(f"{name}: {p:.6g} ± {err:.6g}")
            else:
                formatted.append(f"{name}: {p:.6g}")
        web.page["res-params"].innerHTML = "<br>".join(formatted)
    else:
        web.page["res-params"].textContent = "—"

    web.page["res-notes"].textContent = message if message else "—"


def show_results():
    web.page["results-empty"].classes.add("hidden")
    web.page["results"].classes.remove("hidden")

def optimize(loss, par):
    # Use tighter convergence criteria for better Hessian computation
    res = minimize(loss, par, method="L-BFGS-B", bounds=loss.bounds, jac='3-point',
                   options={'ftol': 1e-12, 'gtol': 1e-8, 'maxiter': 1000})
    #display(res.message)
    
    # Compute Hessian using the LossFunction's method
    hess = loss.compute_hessian(res.x)
    
    # Use pseudoinverse for stable inversion, with regularization if needed
    try:
        cov = np.linalg.pinv(hess, rcond=1e-15)
    except np.linalg.LinAlgError:
        # If inversion fails, add small regularization
        hess_reg = hess + np.eye(len(res.x)) * 1e-10
        cov = np.linalg.pinv(hess_reg, rcond=1e-15)
    
    if loss.loss_type == "least_squares":
        residuals = loss.data[1] - loss.function(loss.data[0], res.x)
        n = len(loss.data[0])
        p = len(par)
        sigma2 = np.sum(residuals**2) / (n - p)
        cov = 2 * cov * sigma2
    elif loss.loss_type == "weighted_least_squares":
        cov = 2 * cov
    
    # Check for NaN or negative variances
    variances = np.diag(cov)
    if np.any(np.isnan(variances)) or np.any(variances < 0):
        # Fallback: use identity matrix scaled by residual variance
        if loss.loss_type in ["least_squares", "weighted_least_squares"]:
            residuals = loss.data[1] - loss.function(loss.data[0], res.x)
            fallback_var = np.var(residuals) if len(residuals) > 1 else 1.0
            cov = np.eye(len(res.x)) * fallback_var
        else:
            cov = np.eye(len(res.x)) * 1e-6  # Small default variance
    
    stderr = np.sqrt(np.maximum(np.diag(cov), 0))  # Ensure non-negative
    return res.x, stderr, res.message


@when("click", "#btn-plot")
def plot():
    plot_type = get_plot_type()
    if plot_type == "scatter":
        plot_scatter()
    elif plot_type == "hist":
        plot_hist()


def plot_scatter():
    global current_fig
    x, y, err = get_data("scatter")
    x_min, x_max = np.min(x), np.max(x)
    x_range = x_max - x_min
    fig1, ax1 = plt.subplots()
    title = web.page["plot-title"].value or " "
    xlabel = web.page["x-label"].value or " "
    ylabel = web.page["y-label"].value or " "
    ax1.set_title(title)
    ax1.set_xlabel(xlabel,fontsize=14)
    ax1.set_ylabel(ylabel,fontsize=14)
    ax1.errorbar(x, y, yerr=err, fmt='o', capsize=3)
    function, parameters = get_function(plottype="scatter")
    if function is not None:
        loss_type = get_loss_type()
        y_pred = function(x, parameters)
        r2, negloglik = calculate_metrics(y, y_pred, err, loss_type)
        web.page["res-negloglik"].textContent = f"{negloglik:.6g}"
        web.page["res-r2"].textContent = f"{r2:.6g}"
        show_results()
        x_plot = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, 100)
        fit_name = web.page["fit-func"].value
        plt.plot(x_plot, function(x_plot, parameters), color='orange', label=f"{fit_name.title()}, R²={r2:.3f}")
        ax1.legend()
    else:
        web.page["results-empty"].classes.remove("hidden")
        web.page["results"].classes.add("hidden")
    plt.tight_layout()
    # Close previous figure to avoid memory leak
    if current_fig is not None:
        plt.close(current_fig)
    current_fig = fig1
    display(fig1, target="mpl", append=False)


def plot_hist():
    global current_fig
    x = get_data("hist")
    fig1, ax1 = plt.subplots()
    title = web.page["plot-title"].value or " "
    xlabel = web.page["x-label"].value or " "
    ylabel = web.page["y-label"].value or " "
    ax1.set_title(title)
    ax1.set_xlabel(xlabel,fontsize=14)
    ax1.set_ylabel(ylabel,fontsize=14)
    bins_val = web.page["hist-bins"].value
    bins = int(bins_val) if bins_val and int(bins_val) >= 1 else None
    ax1.hist(x, **(dict(bins=bins) if bins else {}), density=True)
    function, parameters = get_function(plottype="hist")
    if function is not None:
        probs = function(x, parameters)
        # Use safer minimum probability to avoid numerical underflow
        min_prob = 1e-100
        valid_probs = np.where(np.isfinite(probs) & (probs > 0), np.maximum(probs, min_prob), min_prob)
        gof = -np.sum(np.log(valid_probs))
        web.page["res-negloglik"].textContent = f"{gof:.6g}"
        web.page["res-r2"].textContent = "—"
        show_results()
        x_min, x_max = np.min(x), np.max(x)
        if function is not gaussian:
            x_min = max(x_min, 1e-7)
            x_max = max(x_max, x_min + 0.1)
            x_range = x_max - x_min
            x_plot = np.linspace(x_min, x_max + 0.1 * x_range, 100)
        else:
            x_range = x_max - x_min
            x_plot = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, 100)
        plt.plot(x_plot, function(x_plot, parameters))
    else:
        web.page["results-empty"].classes.remove("hidden")
        web.page["results"].classes.add("hidden")
    plt.tight_layout()
    # Close previous figure to avoid memory leak
    if current_fig is not None:
        plt.close(current_fig)
    current_fig = fig1
    display(fig1, target="mpl", append=False)


@when("click", "#btn-fit")
def fit_data():
    plot_type = get_plot_type()
    if plot_type == "scatter":
        x, y, err = get_data("scatter")
        function, parameters = get_function(plottype="scatter")
        if function is not None:
            loss_type = get_loss_type()
            if loss_type == "wls" and err is not None:
                # Guard against zero/near-zero errors
                err_safe = np.maximum(err, 1e-10)
                loss_function = LossFunction("weighted_least_squares", (x, y, err_safe), function)
            else:
                loss_function = LossFunction("least_squares", (x, y), function)
            opt_par, par_err, message = optimize(loss_function, parameters)
            update_function(opt_par, par_err, message, plottype="scatter")
            plot_scatter()
    elif plot_type == "hist":
        x = get_data("hist")
        function, parameters = get_function(plottype="hist")
        if function is not None:
            loss_function = LossFunction("log_likelihood", [x], function)
            opt_par, par_err, message = optimize(loss_function, parameters)
            update_function(opt_par, par_err, message, plottype="hist")
            plot_hist()


@when("click", "#btn-export")
def export_png():
    global current_fig
    if current_fig is None:
        web.page["action-status"].textContent = "No plot to export. Please plot first."
        return

    buf = io.BytesIO()
    current_fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    data = buf.getvalue()
    encoded = base64.b64encode(data).decode('ascii')

    # Use JS to create and trigger download
    js_code = f"""
    const binaryString = atob('{encoded}');
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {{
        bytes[i] = binaryString.charCodeAt(i);
    }}
    const blob = new Blob([bytes], {{type: 'image/png'}});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'tinyplot.png';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    """
    window.eval(js_code)
    web.page["action-status"].textContent = "Plot exported as 'tinyplot.png'"


### Functions:
def polynomial(x, par):
    res = 0.0
    for i, pi in enumerate(par):
        res += pi * x**i
    return res


def logarithmic(x, par):
    A, b = par
    return A * np.log(x) + b


def exponential(x, par):
    A, k, b = par
    return A * np.exp(-k * x) + b


class LossFunction:
    def __init__(self, loss_type, data, function):
        self.data = data
        self.loss_type = loss_type
        self.function = function
        if self.function == gaussian or self.function == lognormal:
            self.bounds = ((-np.inf, np.inf), (0.0, np.inf))
        elif isinstance(self.function, MaxwellBoltzmann):
            self.bounds = ((0.0, np.inf),)
        elif self.function == logarithmic:
            self.bounds = ((-np.inf, np.inf), (-np.inf, np.inf))
        else:
            self.bounds = None

    def __call__(self, par):
        if self.loss_type == "least_squares":
            residual = self.data[1] - self.function(self.data[0], par)
            loss = np.sum(residual**2)
            return loss
        elif self.loss_type == "weighted_least_squares":
            residual = self.data[1] - self.function(self.data[0], par)
            weights = 1.0 / self.data[2]**2
            loss = np.sum(weights * residual**2)
            return loss
        elif self.loss_type == "log_likelihood":
            probs = self.function(self.data[0], par)
            # Replace any non-positive or non-finite values with small positive value
            valid_probs = np.where(np.isfinite(probs) & (probs > 0), probs, 1e-300)
            loss = -np.sum(np.log(valid_probs))
            return loss

    def compute_hessian(self, par):
        """Compute the Hessian matrix at parameter values par using finite differences"""
        def loss_func(x):
            return self(x)
        
        # Use finite differences to compute Hessian
        eps = 1e-5
        n = len(par)
        hess = np.zeros((n, n))
        
        # Compute Hessian by differentiating the gradient
        for i in range(n):
            def grad_i(x):
                return approx_fprime(x, loss_func, epsilon=eps)[i]
            hess[i, :] = approx_fprime(par, grad_i, epsilon=eps)
        
        # Symmetrize the Hessian to ensure numerical stability
        hess = (hess + hess.T) / 2
        
        return hess


### Distriutions:


# Maxwell-Boltzmann-Distributions
class MaxwellBoltzmann:
    def __init__(self, dim=2):
        self.dim = dim

    def __call__(self, v, par):
        a = par[0]
        if self.dim == 2:
            p = v / a * np.exp(-(v**2) / (2 * a))
        elif self.dim == 3:
            p = np.sqrt(2.0 / np.pi) * v**2 / a**3 * np.exp(-(v**2) / (2 * a**2))
        p[v < 0] = 0.0
        return p


def gaussian(x, par):
    mu, sigma = par
    p = 1.0 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-((x - mu) ** 2) / (2 * sigma**2))
    return p


def lognormal(x, par):
    mu, sigma = par
    p = (
        1.0
        / (np.sqrt(2 * np.pi) * x * sigma)
        * np.exp(-((np.log(x) - mu) ** 2) / (2 * sigma**2))
    )
    # p = (1 / (v*np.sqrt(2 * np.pi*np.log(1+sigma**2/mu**2)))
    #          * np.exp(-((np.log(v) - np.log(mu/np.sqrt(1+sigma**2/mu**2)))**2) /
    #                  (2 * np.log(1+sigma**2/mu**2) )) )
    return p
