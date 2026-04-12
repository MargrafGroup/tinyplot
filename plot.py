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
    rows = [list(r) for r in window.csv_rows][1:]

    def read_col(idx):
        vals = []
        for r in rows:
            if idx < len(r):
                try:
                    v = float(r[idx])
                    if not np.isnan(v):
                        vals.append(v)
                except (ValueError, TypeError):
                    pass
        return np.array(vals, dtype=np.float64)

    def read_cols_aligned(*indices):
        cols = [[] for _ in indices]
        for r in rows:
            try:
                vals = [float(r[idx]) if idx < len(r) else float("nan") for idx in indices]
                if any(np.isnan(v) for v in vals):
                    continue
                for col, v in zip(cols, vals):
                    col.append(v)
            except (ValueError, TypeError):
                pass
        return [np.array(c, dtype=np.float64) for c in cols]

    if plottype == "scatter":
        x_index = int(web.page["x-select"].value)
        y_index = int(web.page["y-select"].value)
        err_select = web.page["err-select"].value
        if err_select and err_select != "" and err_select != "0":
            err_index = int(err_select) - 1
            x, y, err = read_cols_aligned(x_index, y_index, err_index)
        else:
            x, y = read_cols_aligned(x_index, y_index)
            err = None
        return x, y, err
    elif plottype == "hist":
        x_index = int(web.page["hist-column"].value)
        return read_col(x_index)
    elif plottype == "classify":
        a_index = int(web.page["class-col-a"].value)
        b_index = int(web.page["class-col-b"].value)
        return read_col(a_index), read_col(b_index)


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
    elif plottype == "classify":
        func_type = web.page["class-fit"].value
        if func_type == "none":
            return None, (0.0)
        elif func_type == "logistic":
            return logistic_sigmoid, (
                float(web.page["par-logit-w"].value),
                float(web.page["par-logit-b"].value),
            )


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
    elif plottype == "classify":
        func_type = web.page["class-fit"].value
        if func_type == "none":
            return None
        elif func_type == "logistic":
            web.page["par-logit-w"].value = str(parameters[0])
            web.page["par-logit-b"].value = str(parameters[1])

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

    elif plottype == "classify":
        func_type = web.page["class-fit"].value
        if func_type == "logistic":
            model_name = "Logistic Regression"
            equation = r"$$p(y{=}1\mid x) = \frac{1}{1 + e^{-(wx+b)}}$$"
            param_names = ["w", "b"]
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
        if plottype == "hist" and web.page["hist-fit"].value == "lognormal" and len(parameters) == 2:
            mu, sigma = parameters[0], parameters[1]
            linear_mean = np.exp(mu + sigma**2 / 2)
            linear_std = np.sqrt((np.exp(sigma**2) - 1) * np.exp(2 * mu + sigma**2))
            formatted.append(f"mean (linear): {linear_mean:.6g}")
            formatted.append(f"std (linear): {linear_std:.6g}")
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
    elif plot_type == "classify":
        plot_classify()


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


def plot_classify():
    global current_fig
    x_a, x_b = get_data("classify")
    title  = web.page["plot-title"].value or " "
    xlabel = web.page["x-label"].value or " "

    fig, (ax_hist, ax_main) = plt.subplots(
        2, 1, sharex=True,
        gridspec_kw={"height_ratios": [1, 3]},
        figsize=(6, 5)
    )
    fig.suptitle(title)

    # Top panel: overlapping density histograms
    ax_hist.hist(x_a, density=True, alpha=0.5, label="A (y=0)", color="steelblue")
    ax_hist.hist(x_b, density=True, alpha=0.5, label="B (y=1)", color="tomato")
    ax_hist.set_ylabel("Density", fontsize=11)
    ax_hist.legend(fontsize=9)

    # Bottom panel: class scatter
    rng = np.random.default_rng(0)
    jitter_a = rng.uniform(-0.04, 0.04, size=len(x_a))
    jitter_b = rng.uniform(-0.04, 0.04, size=len(x_b))
    ax_main.scatter(x_a, jitter_a,       color="steelblue", alpha=0.5, s=15, label="A (y=0)")
    ax_main.scatter(x_b, 1.0 + jitter_b, color="tomato",    alpha=0.5, s=15, label="B (y=1)")
    ax_main.set_yticks([0, 1])
    ax_main.set_yticklabels(["A (0)", "B (1)"])
    ax_main.set_xlabel(xlabel, fontsize=12)
    ax_main.set_ylabel("Class", fontsize=12)

    function, parameters = get_function(plottype="classify")
    if function is not None:
        x_all = np.concatenate([x_a, x_b])
        y_all = np.concatenate([np.zeros(len(x_a)), np.ones(len(x_b))])
        p = function(x_all, parameters)
        p = np.clip(p, 1e-12, 1 - 1e-12)
        negloglik = -np.sum(y_all * np.log(p) + (1 - y_all) * np.log(1 - p))
        web.page["res-negloglik"].textContent = f"{negloglik:.6g}"
        web.page["res-r2"].textContent = "—"
        show_results()

        x_range = np.max(x_all) - np.min(x_all)
        x_plot = np.linspace(np.min(x_all) - 0.1 * x_range, np.max(x_all) + 0.1 * x_range, 200)
        ax_main.plot(x_plot, function(x_plot, parameters), color="darkorange", lw=2, label="Logistic fit")
        ax_main.legend(fontsize=9)
    else:
        web.page["results-empty"].classes.remove("hidden")
        web.page["results"].classes.add("hidden")

    plt.tight_layout()
    if current_fig is not None:
        plt.close(current_fig)
    current_fig = fig
    display(fig, target="mpl", append=False)


@when("click", "#btn-fit")
def fit_data():
    plot_type = get_plot_type()
    if plot_type == "scatter":
        x, y, err = get_data("scatter")
        function, parameters = get_function(plottype="scatter")
        if function is not None:
            loss_type = get_loss_type()
            func_type = web.page["fit-func"].value

            # --- Logarithmic: exact linear solve, no optimizer ---
            if func_type == "logarithmic":
                u = np.log(np.maximum(x, 1e-10))
                X = np.column_stack([u, np.ones_like(u)])
                w = 1.0 / np.maximum(err, 1e-10)**2 if (loss_type == "wls" and err is not None) else None
                opt_par, par_err = linear_lstsq(X, y, w=w)
                message = "Exact linear solution (log transform)"
                update_function(opt_par, par_err, message, plottype="scatter")
                plot_scatter()
                return

            # --- Exponential: log-space initializer ---
            if func_type == "exponential":
                mask = y > 0
                if np.sum(mask) >= 2:
                    u = np.log(y[mask])
                    X_lin = np.column_stack([x[mask], np.ones(np.sum(mask))])
                    # In log space: sigma_ln(y) ≈ sigma_y / y, so w_log = (y/sigma_y)^2
                    w_lin = (y[mask] / np.maximum(err[mask], 1e-10))**2 if (loss_type == "wls" and err is not None) else None
                    lin_par, _ = linear_lstsq(X_lin, u, w=w_lin)
                    k0, alpha0 = lin_par
                    A0 = np.exp(np.clip(alpha0, -500, 500))
                    parameters = (A0, k0, 0.0)

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
    elif plot_type == "classify":
        x_a, x_b = get_data("classify")
        function, parameters = get_function(plottype="classify")
        if function is not None:
            x_all = np.concatenate([x_a, x_b])
            y_all = np.concatenate([np.zeros(len(x_a)), np.ones(len(x_b))])
            loss_function = LossFunction("logistic", (x_all, y_all), function)
            opt_par, par_err, message = optimize(loss_function, parameters)
            update_function(opt_par, par_err, message, plottype="classify")
            plot_classify()


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


def linear_lstsq(X, y, w=None):
    """Weighted least-squares solve: X @ params = y.
    Returns (params, stderr) where cov = sigma2 * (X^T W X)^-1.
    """
    if w is not None:
        sw = np.sqrt(w)
        Xw = X * sw[:, None]
        yw = y * sw
    else:
        Xw, yw = X, y
    params, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
    residuals = y - X @ params
    n, p = X.shape
    sigma2 = np.sum((residuals if w is None else residuals * w) * residuals) / max(n - p, 1)
    try:
        cov = sigma2 * np.linalg.inv(Xw.T @ Xw)
        stderr = np.sqrt(np.maximum(np.diag(cov), 0))
    except np.linalg.LinAlgError:
        stderr = np.full(p, np.nan)
    return params, stderr


def logarithmic(x, par):
    A, b = par
    return A * np.log(np.maximum(x, 1e-10)) + b


def exponential(x, par):
    A, k, b = par
    return A * np.exp(np.clip(k * x, -500, 500)) + b


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
        elif self.function == exponential:
            x_max_abs = np.max(np.abs(self.data[0]))
            k_bound = 500.0 / max(x_max_abs, 1e-10)
            self.bounds = ((-np.inf, np.inf), (-k_bound, k_bound), (-np.inf, np.inf))
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
        elif self.loss_type == "logistic":
            x, y = self.data
            p = self.function(x, par)
            p = np.clip(p, 1e-12, 1 - 1e-12)
            loss = -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
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


def logistic_sigmoid(x, par):
    w, b = par
    z = w * x + b
    # Numerically stable sigmoid
    p = np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))
    return p
