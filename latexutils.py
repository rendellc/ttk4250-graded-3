import csv
from numpy import pi, rad2deg

PARAMETER_TO_TEXNAME = dict(
    sigma_range = r"\sigma_r",
    alpha_joint = r"\alpha_\text{joint}",
    alpha_individual = r"\alpha_\text{individual}",
)

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

def parameter_to_texvalues(params):
    parameter_tex_dict = {}
    for paramname,value in params.items():
        if paramname in PARAMETER_TO_TEXNAME.keys():
            tex_key = PARAMETER_TO_TEXNAME[paramname]
            parameter_tex_dict[tex_key] = value

    p = params
    parameter_tex_dict[r"\sigma_\theta"] = rad2deg(p["sigma_bearing"])

    return parameter_tex_dict


def save_params_to_csv(params, filename, headers=[]):
    #texparameters = parameter_to_texvalues(params)

    with open(filename, 'w', newline='') as csvfile:
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

    round_str = lambda num: round(num,3)
    percent_str = lambda num: rf"{(100*num):.1f}\%"

    with open(filename, 'w', newline='') as csvfile:
        print("Writing consistency results to", csvfile.name)
        writer = csv.writer(csvfile, delimiter=';',quoting=csv.QUOTE_MINIMAL)

        # write so that latex will interpret it correctly
        # cant start a line with backslash so first column is empty

        #writer.writerow(["", "", "Inside", "Averaged", "CI"])
        for consdata in consdatas:
            avg = round_str(consdata["avg"])
            inside = percent_str(consdata["inside"])
            text = consdata["text"]

            CI = consdata.get("CI", [])
            if len(CI) > 0:
                CItext = f"({round_str(CI[0])},{round_str(CI[1])})"
            else:
                CItext = "-"

            writer.writerow(["", text, inside, avg, CItext])

def save_value(name, value, filename):
    with open(filename, 'w', newline='') as csvfile:
        print(f"Saving {name} to", csvfile.name)
        writer = csv.writer(csvfile, delimiter=';',quoting=csv.QUOTE_MINIMAL)
        writer.writerow([name, value])


