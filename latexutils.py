import csv
from numpy import pi, rad2deg

from pathlib import Path

PARAMETER_TO_TEXNAME = dict(
    sigma_x = r"\sigma_\text{x}",
    sigma_y = r"\sigma_\text{y}",
    sigma_range = r"\sigma_r",
    alpha_joint = r"\alpha_\text{joint}",
    alpha_individual = r"\alpha_\text{individual}",
)

SAVE_DIR = Path(".")

def set_save_dir(dirname):
    """
    Set the default save directory for module.
    """
    global SAVE_DIR
    SAVE_DIR = Path(dirname)

    if not SAVE_DIR.exists():
        print("Creating", SAVE_DIR)
        SAVE_DIR.mkdir()

def sig_exp(num_str):
    parts = num_str.split('.', 2)
    decimal = parts[1] if len(parts) > 1 else ''
    exp = -len(decimal)
    digits = parts[0].lstrip('0') + decimal
    trimmed = digits.rstrip('0')
    exp += len(digits) - len(trimmed)
    sig = int(trimmed) if trimmed else 0
    return sig, exp

def sig_exp_e(num_str):
    v, exp = num_str.split('e')
    return float(v), int(exp)

def _tentothepower(paramname, paramdict):
    p = paramdict
    pt = PARAMETER_TO_TEXNAME

    v, exp = sig_exp_e(str(p[paramname]))
    v_str = "" if abs(v-1) < 0.0001 else str(v)
    return v_str +" 10^{" + str(exp) + "}"


def parameter_to_texvalues(params):
    parameter_tex_dict = {}
    for paramname,value in params.items():
        if paramname in PARAMETER_TO_TEXNAME.keys():
            tex_key = PARAMETER_TO_TEXNAME[paramname]
            parameter_tex_dict[tex_key] = value

    p = params
    pt = PARAMETER_TO_TEXNAME

    parameter_tex_dict[r"\sigma_\psi"] = f"{rad2deg(p['sigma_psi'])}" + r" \text{deg}"

    parameter_tex_dict[r"\sigma_\theta"] = f"{rad2deg(p['sigma_bearing'])}" + r" \text{deg}"

    parameter_tex_dict[pt["alpha_individual"]] = _tentothepower("alpha_individual", p)
    parameter_tex_dict[pt["alpha_joint"]] = _tentothepower("alpha_joint", p)


    return parameter_tex_dict


def save_params_to_csv(params, filename, headers=[]):
    #texparameters = parameter_to_texvalues(params)
    p = SAVE_DIR / filename

    with p.open(mode='w', newline='') as csvfile:
        print("Writing parameters to", csvfile.name)
        writer = csv.writer(csvfile, delimiter=';',quoting=csv.QUOTE_MINIMAL)

        # write so that latex will interpret it correctly
        # cant start a line with backslash so first column is empty
        if headers:
            writer.writerow(["", *headers])
        for k,v in params.items():
            writer.writerow(["", k, v])


def save_consistency_results(consdatas, filename):
    # consdatas : [{avg, inside, text, CI}]
    p = SAVE_DIR / filename

    round_str = lambda num: f"{num:.3f}"
    percent_str = lambda num: rf"{(100*num):.1f}\%"

    with p.open(mode='w', newline='') as csvfile:
        print("Writing consistency results to", csvfile.name)
        writer = csv.writer(csvfile, delimiter=';',quoting=csv.QUOTE_MINIMAL)

        # write so that latex will interpret it correctly
        # cant start a line with backslash so first column is empty
        # % must be escaped

        #writer.writerow(["", "", "Inside", "Averaged", "CI"])
        for consdata in consdatas:
            if "avg" in consdata:
                avg = round_str(consdata["avg"])
            else:
                avg = ""

            inside = percent_str(consdata["inside"])
            text = consdata["text"]

            CI = consdata.get("CI", [])
            if len(CI) > 0:
                CItext = f"({round_str(CI[0])},{round_str(CI[1])})"
            else:
                CItext = ""

            writer.writerow(["", text, inside, avg, CItext])

def save_value(name, value, filename):
    p = SAVE_DIR / filename
    with p.open(mode='w', newline='') as csvfile:
        print(f"Saving {name} to", csvfile.name)
        writer = csv.writer(csvfile, delimiter=';',quoting=csv.QUOTE_MINIMAL)
        writer.writerow([name, value])

def save_fig(fig, filename):
    p = SAVE_DIR / filename
    fig.savefig(p)

