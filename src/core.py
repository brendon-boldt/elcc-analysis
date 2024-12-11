from typing import Literal, Any
from pathlib import Path
import json
import itertools

from matplotlib import pyplot as plt
import seaborn as sns  # type: ignore
import pandas as pd
import numpy as np
import matplotlib

sns.set_theme()
sns.set_context("paper")

OUTPUT_PATH = Path("./analysis-output")

log_scale_cols = [
    "Entropy per Line",
    "Line Count",
    "Token Count",
    "Tokens per Line",
    "Unique Lines",
    "Unique Tokens",
]


def get_xferbench_output(
    which: Literal["elcc", "var", "all"] = "elcc",
) -> pd.DataFrame:
    records = []

    results_paths = itertools.chain.from_iterable(
        Path("xferbench-output/").glob(f"{pat}/*/results.json")
        for pat in ["full-*", "var-lang", "var-xb"]
    )
    for p in results_paths:
        if p.parents[0].name == "xferbench-no-pretrain":
            system = "no-pretrain"
            variant = "no-pretrain"
        else:
            components = p.parents[0].name.split("_")[1:]
            system = components[0]
            variant = "_".join(components[1:])

        with p.open() as fo:
            data = json.load(fo)

        system_path = Path(f"elcc/systems/{system}")
        corpus_path = system_path / f"data/{variant}/metadata.json"

        is_baseline = not system_path.exists()
        if system in ("ar", "es", "fr", "hi", "ko", "ru", "zh"):
            variant = system
            system = "hl"

        match system:
            case "hl":
                lang_type = "human"
            case "rand":
                lang_type = "synthetic"
            case "no-pretrain":
                lang_type = "baseline"
            case "pz" | "parens":
                lang_type = "synthetic"
            case _:
                lang_type = "emergent"

        target_scores = {f"target_{k}": v for k, v in data["by_target"].items()}
        record = {
            "path": str(p),
            "system": system,
            "variant": variant,
            "type": lang_type,
            "score": data["score"],
            "baseline": is_baseline,
            "run_id": p.parents[-3].name.split("-")[-1],
            **target_scores,
            **data.get("analysis", {}),
        }
        if not is_baseline:
            with (system_path / "system.json").open() as fo:
                system_metadata = json.load(fo)
            with corpus_path.open() as fo:
                corpus_metadata = json.load(fo)

            system_metrics = corpus_metadata["metrics"].get("system", {})
            success_fields = [
                "win_vil_mean",
                "ensemble_acc",
                "acc",  # Out of 100
                "success_rate",
            ]
            success = None
            for f in success_fields:
                if f in system_metrics:
                    success = system_metrics[f]
                    if f == "acc":
                        success /= 100
                    continue

            record.update(
                {
                    **system_metadata["system"],
                    "success": success,
                }
            )
        records.append(record)

    xbo = pd.DataFrame.from_records(records, index=["system", "variant", "run_id"])

    center = xbo.loc["no-pretrain"]["score"].mean()
    xbo["score_center"] = xbo["score"] - center

    for col in log_scale_cols:
        xbo["Log " + col] = np.log10(xbo[col])

    match which:
        case "elcc":
            path_re = r"full-[0-9]+/"
        case "var":
            path_re = r"var-(lang|xb)"
        case "all":
            path_re = r".*"

    path_filter = xbo.path.str.match(f"xferbench-output/{path_re}")
    xbo = xbo.loc[path_filter]

    return xbo


def mu_goodman_bar() -> None:
    sns.set_context("notebook", font_scale=1.2)
    df = get_xferbench_output("elcc")
    df = df.reset_index()
    df = df.loc[df.system == "generalizations-mu-goodman"]
    df = df.rename(columns={"variant": "Variant", "score": "XferBench Score"})
    df["Variant"] = df["Variant"].replace(
        {
            "cub": "CUB",
            "shapeworld": "SW",
            "reference": "Ref",
            "concept": "Concept",
            "set_": "Set ",
            "-": ", ",
        },
        regex=True,
    )

    def sorter(x) -> tuple[int, int]:
        match x.split(" ")[1]:
            case "Concept":
                y = 0
            case "Set":
                y = 1
            case "Ref":
                y = 2
        return x.split(" ")[0], y

    df["key"] = df["Variant"].apply(sorter)
    df = df.sort_values("key")

    plot = sns.barplot(
        df,
        x="XferBench Score",
        y="Variant",
    )
    plot.set_xlim([6.01, 6.050])

    plot.figure.tight_layout()
    plot.figure.savefig(OUTPUT_PATH / "mu-goodman.png")
    plt.close()


def entropy_scatter() -> None:
    sns.set_context("notebook", font_scale=1.2)
    df = get_xferbench_output("elcc")
    df = df.loc[df.type.isin(["human", "emergent"])]
    df["type"] = df["type"].replace({"human": "Human", "emergent": "Emergent"})
    df = df.rename(columns={"score": "XferBench Score", "type": "Type"})

    plot = sns.scatterplot(
        df,
        y="XferBench Score",
        x="1-gram Entropy",
        hue="Type",
        style="Type",
        alpha=0.5,
    )
    plot.figure.tight_layout()
    plot.figure.savefig(OUTPUT_PATH / "entropy-scatter.png")
    plt.close()


def proposal_entropy_scatter() -> None:
    sns.set_context("notebook", font_scale=1.2)
    df = get_xferbench_output("elcc")
    df = df.loc[df.type.isin(["human", "emergent"])]
    df["type"] = df["type"].replace({"human": "Human", "emergent": "Emergent"})
    df = df.rename(columns={"score": "XferBench Score", "type": "Type"})

    plot = sns.scatterplot(
        df,
        y="XferBench Score",
        x="1-gram Entropy",
        hue="Type",
        style="Type",
        alpha=0.5,
    )
    plot.figure.tight_layout()
    plot.figure.savefig("analysis-output/proposal-entropy-scatter.png")
    plt.close()


def success_scatter() -> None:
    sns.set_context("notebook", font_scale=1.2)
    df = get_xferbench_output("elcc")
    df = df.loc[df.type == "emergent"]
    df = df.reset_index()
    df = df.loc[~df.success.isna()]
    df = df.rename(
        columns={
            "score": "XferBench Score",
            "success": "Success Rate",
            "type": "Type",
            "system": "System",
        }
    )

    plot = sns.lmplot(
        df,
        y="XferBench Score",
        x="Success Rate",
        hue="System",
        ci=None,
        scatter_kws=dict(
            alpha=0.5,
        ),
    )
    sns.move_legend(plot, loc="upper right", title=None, bbox_to_anchor=(0.5, 0.96))
    plot.figure.tight_layout()
    # plt.legend(loc="upper left")
    plot.figure.savefig(OUTPUT_PATH / "success-scatter.png")
    plt.close()


def ec_at_scale_bar() -> None:
    sns.set_context("notebook", font_scale=1.2)
    df = get_xferbench_output("elcc")
    df = df.reset_index()
    df = df.rename(columns={"variant": "Variant", "score": "XferBench Score"})
    df["Variant"] = df["Variant"].replace(
        {
            "imagenet-": "",
            "x": "S, ",
            "$": "R",
        },
        regex=True,
    )
    df = df.loc[df.system == "ec-at-scale"]

    plot = sns.barplot(
        df,
        x="XferBench Score",
        y="Variant",
    )
    plot.set_xlim([6.02, 6.045])

    plot.figure.tight_layout()
    plot.figure.savefig(OUTPUT_PATH / "ec-at-scale.png")
    plt.close()


def elcc_paper_cat() -> None:
    sns.set_context("notebook", font_scale=1.5)
    pd.options.display.float_format = "{:.2f}".format

    df = get_xferbench_output()
    df.reset_index(inplace=True)
    npt_mean = df.loc[df.system == "no-pretrain", "score"].mean()
    df = df.rename(columns={"system": "System", "score": "XferBench Score"})
    systems = {
        "hl": "Human Language",
        "no-pretrain": "No Pretrain",
        "egg-discrimination": "Signal, discrimination",
        "egg-reconstruction": "Signal, reconstruction",
        "corpus-transfer-yao-et-al": "Signal, natural images",
        "generalizations-mu-goodman": "Signal, concept-based",
        "ec-at-scale": "Signal, population",
        "babyai-sr": "Navigation, gridworld",
        "nav-to-center": "Navigation, continuous",
        "rlupus": "Social deduction, RLupus",
    }
    df.System = df.System.replace(systems)
    df = df.loc[~df.System.isin(["rand", "pz"])]

    def sorter(x: str) -> Any:
        return list(systems.values()).index(x)

    df["key"] = df.System.apply(sorter)
    df = df.sort_values("key")

    plot = sns.catplot(
        df,
        y="System",
        x="XferBench Score",
        dodge=True,
        alpha=0.3,
        height=5,
        aspect=2,
    )
    plot.fig.tight_layout()
    plt.plot([npt_mean, npt_mean], [-1, 10], alpha=0.3)
    plot.axes[0, 0].set_ylim([10, -1])
    plot.fig.savefig(OUTPUT_PATH / "elcc-cat.png")
    plt.close()


def generate_elcc_paper_output() -> None:
    matplotlib.use("agg")
    OUTPUT_PATH.mkdir(exist_ok=True, parents=True)

    # General cat plot of ELCC contents
    elcc_paper_cat()
    # Entropy vs XferBench scatter plot
    entropy_scatter()
    # Success vs XferBench scatter plot
    success_scatter()
    # Generalizations bar plot
    mu_goodman_bar()
    # EC at Scale scatter plot
    ec_at_scale_bar()

def export_xferbench_output() -> None:
    df = get_xferbench_output()
    df.to_csv("./xferbench-output.csv")
