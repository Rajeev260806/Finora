import matplotlib.pyplot as plt


def plot_portfolio_value(histories, output_file="portfolio_value_comparison.png"):
    plt.figure(figsize=(12, 6))

    for name, history in histories.items():
        plt.plot(
            history.index,
            history["portfolio_value"],
            label=name
        )

    plt.axhline(
        y=1.0,
        linestyle="--",
        label="Starting Value"
    )

    plt.title("Portfolio Value Comparison")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(output_file)
    plt.close()


def plot_drawdown(histories, output_file="drawdown_comparison.png"):
    plt.figure(figsize=(12, 6))

    for name, history in histories.items():
        portfolio = history["portfolio_value"]
        running_max = portfolio.cummax()
        drawdown = portfolio / running_max - 1

        plt.plot(
            drawdown.index,
            drawdown,
            label=name
        )

    plt.axhline(
        y=0,
        linestyle="--"
    )

    plt.title("Drawdown Comparison")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(output_file)
    plt.close()


def plot_turnover(histories, output_file="turnover_comparison.png"):
    plt.figure(figsize=(12, 6))

    for name, history in histories.items():
        if "turnover" in history.columns:
            plt.plot(
                history.index,
                history["turnover"],
                label=name
            )

    plt.title("Turnover Comparison")
    plt.xlabel("Date")
    plt.ylabel("Turnover")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(output_file)
    plt.close()


def plot_return_distribution(
    returns,
    output_file="return_distribution.png"
):
    plt.figure(figsize=(12, 6))

    for name, values in returns.items():
        plt.hist(
            values.dropna(),
            bins=50,
            alpha=0.5,
            label=name
        )

    plt.title("Daily Return Distribution")
    plt.xlabel("Daily Return")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(output_file)
    plt.close()


def plot_final_values(
    comparison_df,
    output_file="strategy_comparison.png"
):
    plt.figure(figsize=(9, 5))

    plt.bar(
        comparison_df.index,
        comparison_df["Final Portfolio Value"]
    )

    plt.title("Final Portfolio Value Comparison")
    plt.xlabel("Strategy")
    plt.ylabel("Final Portfolio Value")

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()