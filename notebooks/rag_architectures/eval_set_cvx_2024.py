"""Gold questions for EDGAR_ARS_CVX_2024.pdf, embedded into each notebook by build_notebooks.py.

`pages` are 1-indexed PDF pages (printed page number + 2). `verify_eval_set.py` checks every answer against them.
"""

PDF_NAME = "EDGAR_ARS_CVX_2024.pdf"
PDF_SHA256 = "e3b035cfdf2f079ea812da5340a12bd4941229cb2ce4e12d949cbe869c0d9cb2"
MILLION = 1_000_000


def _n(qid, kind, question, value, pages, unit, scale=1):
    return {"id": qid, "type": "numeric", "kind": kind, "question": question, "value": value, "scale": scale,
            "pages": pages, "ground_truth": f"{value} {unit} [p.{pages[0]}]"}


def _t(qid, question, keywords, answer, pages):
    return {"id": qid, "type": "text", "kind": "text", "question": question, "keywords": keywords, "pages": pages,
            "ground_truth": f"{answer} [p.{pages[0]}]"}


def _x(qid, question):
    return {"id": qid, "type": "negative", "kind": "negative", "question": question, "pages": [],
            "ground_truth": "This report does not contain this information."}


USD_M = "million US dollars"
EVAL_SET = [
    # Tables: financial and operating highlights, MD&A, statements.
    _n("T01", "table", "What was Chevron's net income attributable to Chevron Corporation in 2024?", "17,661", [22, 27], USD_M, MILLION),
    _n("T02", "table", "What were Chevron's sales and other operating revenues in 2023?", "196,913", [22, 27, 56], USD_M, MILLION),
    _n("T03", "table", "What was Chevron's cash flow from operating activities in 2022?", "49,602", [22, 59], USD_M, MILLION),
    _n("T04", "table", "What were Chevron's capital expenditures in 2024?", "16,448", [22, 59], USD_M, MILLION),
    _n("T05", "table", "What were Chevron's total assets at year-end 2023?", "261,632", [22, 58], USD_M, MILLION),
    _n("T06", "table", "What was Chevron Corporation stockholders' equity at year-end 2024?", "152,318", [22, 58], USD_M, MILLION),
    _n("T07", "table", "What were diluted earnings per share in 2024?", "9.72", [22, 27], "US dollars per share"),
    _n("T08", "table", "What were cash dividends per share in 2023?", "6.04", [22, 27], "US dollars per share"),
    _n("T09", "table", "What was Chevron's net debt ratio in 2024?", "10.4", [22], "percent"),
    _n("T10", "table", "What was the return on average capital employed in 2022?", "20.3", [22, 27], "percent"),
    _n("T11", "table", "What was net production of crude oil, condensate and synthetic oil in 2024, in thousands of barrels per day?", "1,560", [23], "thousand barrels per day"),
    _n("T12", "table", "What was net production of natural gas liquids in 2022, in thousands of barrels per day?", "279", [23], "thousand barrels per day"),
    _n("T13", "table", "What was net production of natural gas in 2023, in millions of cubic feet per day?", "7,744", [23], "million cubic feet per day"),
    _n("T14", "table", "What were net proved oil-equivalent reserves at year-end 2024, in millions of barrels?", "9,804", [23], "million barrels"),
    _n("T15", "table", "How many employees did Chevron have at year-end 2024, excluding service station employees?", "39,742", [23], "employees"),
    _n("T16", "table", "How many service station employees were excluded from the 2024 employee count?", "5,556", [23], "service station employees"),
    _n("T17", "table", "What were total upstream earnings in 2024?", "18,602", [27], USD_M, MILLION),
    _n("T18", "table", "What were United States downstream earnings in 2023?", "3,904", [27], USD_M, MILLION),
    _n("T19", "table", "What was the All Other earnings line in 2022?", "2,974", [27], USD_M + " (a loss)", MILLION),
    _n("T20", "table", "What was goodwill on the balance sheet at December 31, 2024?", "4,578", [58], USD_M, MILLION),
    _n("T21", "table", "What were total liabilities at December 31, 2024?", "103,781", [58], USD_M, MILLION),
    _n("T22", "table", "What was long-term debt at December 31, 2023?", "20,307", [58], USD_M, MILLION),
    _n("T23", "table", "What was depreciation, depletion and amortization in the 2024 cash flow statement?", "17,282", [59], USD_M, MILLION),
    _n("T24", "table", "What was net cash used for investing activities in 2023?", "15,232", [59], USD_M, MILLION),
    _n("T25", "table", "What were total revenues and other income in 2024?", "202,792", [56], USD_M, MILLION),
    _n("T26", "table", "What was the effective income tax rate in 2024?", "35.5", [79], "percent"),
    # Narrative figures and stat tiles laid over photographs.
    _n("N01", "narrative", "What did Permian Basin production average in 2024, in net barrels of oil-equivalent per day?", "921,000", [7], "net boe per day"),
    _n("N02", "narrative", "Up to what pressure can the Anchor project technology operate, in pounds per square inch?", "20,000", [2, 7], "pounds per square inch"),
    _n("N03", "narrative", "How many gross wells were added to Chevron's Gulf of America production in 2024?", "17", [2], "gross wells"),
    _n("N04", "narrative", "What is the design capacity of the Anchor FPU in gross barrels of oil per day?", "75,000", [3], "gross barrels of oil per day"),
    _n("N05", "narrative", "How many thousand net barrels of oil-equivalent per day came from Chevron's Gulf of America assets in 2024?", "192", [2], "thousand net boe per day"),
    _n("N06", "narrative", "What total assets figure, in billions of dollars, does the Chevron at a glance page show?", "256.9", [18], "billion US dollars"),
    # Charts: reading these needs the picture, because the labels are not tied to a series in the text.
    _n("C01", "chart", "In the five-year cumulative total returns graph, what was $100 invested in Chevron stock worth at the end of 2024?", "150", [21], "US dollars"),
    _n("C02", "chart", "In the five-year cumulative total returns graph, what was $100 invested in the S&P 500 worth at the end of 2024?", "197", [21], "US dollars"),
    _n("C03", "chart", "In the five-year cumulative total returns graph, what was $100 invested in the peer group worth at the end of 2024?", "153", [21], "US dollars"),
    _n("C04", "chart", "According to the total stockholder returns bar chart, what was Chevron's 1-year annualized return as of 12/31/2024?", "1.4", [20], "percent"),
    _n("C05", "chart", "According to the total stockholder returns bar chart, what was Chevron's 5-year annualized return as of 12/31/2024?", "8.5", [20], "percent"),
    _n("C06", "chart", "According to the total stockholder returns bar chart, what was Chevron's 10-year annualized return as of 12/31/2024?", "7.0", [20], "percent"),
    _n("C07", "chart", "What S&P 500 10-year return does the total stockholder returns chart mark?", "13.1", [20], "percent"),
    _n("C08", "chart", "What compound annual growth rate does the indexed dividend growth chart show for Chevron?", "6.2", [20], "percent"),
    # Plain text facts.
    _t("F01", "Who is Chevron's Chairman of the Board and Chief Executive Officer?", ["wirth"], "Michael K. (Mike) Wirth", [12]),
    _t("F02", "Who is Chevron's Lead Director?", ["austin"], "Wanda M. Austin", [12]),
    _t("F03", "Which independent registered public accounting firm audited Chevron's financial statements?", ["pricewaterhouse"], "PricewaterhouseCoopers LLP", [53, 55]),
    _t("F04", "What rating does Standard and Poor's give the bonds issued or guaranteed by Chevron Corporation?", ["aa-"], "AA-", [41]),
    _t("F05", "Which companies are in the peer group used in the stock performance charts?", ["bp", "exxon", "shell", "total"], "BP, ExxonMobil, Shell and TotalEnergies", [20, 21]),
    _t("F06", "Which company did Chevron agree to acquire in an all-stock merger announced in October 2023?", ["hess"], "Hess Corporation", [95]),
    # Unanswerable. Several sit close to something the report does contain.
    _x("X01", "How much bitcoin does Chevron hold on its balance sheet?"),
    _x("X02", "What was Tesla's total revenue in 2024?"),
    _x("X03", "What was ExxonMobil's net income in 2024?"),
    _x("X04", "What was Apple Inc.'s revenue in 2024?"),
    _x("X05", "What was Chevron's net income in 2021?"),
    _x("X06", "What were Chevron's sales and other operating revenues in 2020?"),
    _x("X07", "What was Chevron's net income in 2025?"),
    _x("X08", "What capital expenditures did Chevron report for 2018?"),
    _x("X09", "What was Michael Wirth's total compensation in 2024?"),
    _x("X10", "How many electric vehicle charging stations does Chevron operate?"),
    _x("X11", "What percentage of Chevron's employees are women?"),
    _x("X12", "How many solar panels did Chevron install in 2024?"),
    _x("X13", "What was Chevron's closing share price on December 31, 2024?"),
    _x("X14", "What was the gross margin on Chevron's lithium business in 2024?"),
    _x("X15", "What is the home address of Chevron's Chief Financial Officer?"),
    _x("X16", "What is Chevron's forecast net income for 2027?"),
    _x("X17", "How many offshore wind turbines did Chevron build in 2024?"),
    _x("X18", "How many retail coffee shops does Chevron own in Japan?"),
    _x("X19", "What was Chevron's employee attrition rate in 2024?"),
    _x("X20", "What was Shell's total production in 2024?"),
]
