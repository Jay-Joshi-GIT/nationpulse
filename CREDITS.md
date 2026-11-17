# Credits, data sources and licences

NationPulse compares an explainable model (XGBoost + SHAP) with a large language
model (Mistral 7B) on scoring countries' economies and proposing strategies.
It is built on public data, open-source software and published research. This
file says where each piece comes from, under which terms it may be used, and how
to cite it.

None of the organisations below endorse this project. Their names are used only
to credit their work.

---

## 1. Data

The data was downloaded on 20 October 2026 by `data_pipeline.py`, for 52
countries and the years 2014–2023.

### World Bank: World Development Indicators (WDI)

Seven indicators and each country's name, region and income group come from the
World Bank API (`api.worldbank.org`).

- **Licence:** Creative Commons Attribution 4.0 (CC BY 4.0), with one added
  condition: users may not suggest that the World Bank endorses their work or use
  its name or logo for that purpose.
  [World Bank Terms of Use for Datasets](https://www.worldbank.org/en/about/legal/terms-of-use-for-datasets)
- **Cite as:** World Bank (2026). *World Development Indicators.* Washington, DC:
  The World Bank. https://databank.worldbank.org/source/world-development-indicators
  (accessed 20 October 2026). Licence: CC BY 4.0.

The World Bank compiles some indicators from other organisations, which should
be credited too. The World Bank's own indicator records name these sources:

| Column in this project | WDI indicator (code) | Original source | That source's terms |
|---|---|---|---|
| `gdp_per_capita` (used for the score) | GDP per capita, current US$ (NY.GDP.PCAP.CD) | National statistical offices and central banks; OECD National Accounts; World Bank staff estimates | Through WDI, CC BY 4.0 |
| `inflation` | Inflation, consumer prices, annual % (FP.CPI.TOTL.ZG) | International Monetary Fund, International Financial Statistics | [IMF Copyright and Usage](https://www.imf.org/en/About/copyright-and-terms) |
| `unemployment` | Unemployment, % of labour force, modelled ILO estimate (SL.UEM.TOTL.ZS) | International Labour Organization, ILO Modelled Estimates (ILOSTAT) | CC BY 4.0 for ILO datasets since 3 May 2023 ([ILO rights and permissions](https://www.ilo.org/node/9321)) |
| `foreign_direct_investment` | FDI, net inflows, % of GDP (BX.KLT.DINV.WD.GD.ZS) | IMF International Financial Statistics and Balance of Payments; World Bank; OECD | IMF terms, as above |
| `trade_openness` | Trade, % of GDP (NE.TRD.GNFS.ZS) | National statistical offices and central banks; OECD National Accounts; World Bank staff estimates | Through WDI, CC BY 4.0 |
| `human_development` (life expectancy) | Life expectancy at birth, years (SP.DYN.LE00.IN) | United Nations, DESA Population Division, *World Population Prospects*; national statistical offices; Eurostat | CC BY 3.0 IGO (stated in UN DESA's WPP publications) |
| `education_spending` | Government expenditure on education, % of GDP (SE.XPD.TOTL.GD.ZS) | UNESCO Institute for Statistics (UIS) | CC BY-SA 4.0 ([UIS terms](https://databrowser.uis.unesco.org/terms-and-conditions)), "ShareAlike" |

Original sources, for a full reference list:

- International Labour Organization. *ILO Modelled Estimates (ILOEST)*, ILOSTAT. https://ilostat.ilo.org/
- United Nations, Department of Economic and Social Affairs, Population Division (2024). *World Population Prospects 2024.* https://population.un.org/wpp/
- UNESCO Institute for Statistics. *UIS Data Browser.* https://databrowser.uis.unesco.org/
- International Monetary Fund. *International Financial Statistics* and *Balance of Payments Statistics.* https://data.imf.org/

### International Monetary Fund: government debt

`government_debt` is general government gross debt as a % of GDP
(indicator GGXWDG_NGDP), from the IMF DataMapper API, whose source is the
*World Economic Outlook* database. The World Bank has no debt data for many
countries, so the IMF was used.

- **Terms:** IMF statistical data, including the WEO database, may be reused free
  of charge as long as the IMF is named as the source. **Commercial reuse needs
  the IMF's permission** (copyright@imf.org).
  [IMF Copyright and Usage](https://www.imf.org/en/About/copyright-and-terms)
- **Cite as:** International Monetary Fund (2026). *World Economic Outlook
  Database, October 2026: General government gross debt (GGXWDG_NGDP).* IMF
  DataMapper. https://www.imf.org/external/datamapper/GGXWDG_NGDP (accessed
  20 October 2026).

### What this means for the data files in this repository

`economic_data.csv` and the files made from it (`xgb_scores.csv`,
`shap_scores.csv`, `comparison.csv`, `conformal_bounds.csv`, `strategy_*.csv`,
`scoreboard.csv` and the `results/` folder) contain or are built from the data
above. **This data belongs to its providers, not to this project.** Anyone
reusing them must follow each source's terms:

1. Credit every source listed above.
2. The education spending figures come from UNESCO UIS under CC BY-SA 4.0. If
   you share a changed version of that data, share it under the same licence.
3. The IMF figures (government debt, and through the World Bank, inflation and
   FDI) need the IMF's permission for commercial use.
4. Don't suggest that the World Bank, IMF, ILO, UN or UNESCO endorse your work.

---

## 2. The language model

| Software | Made by | Licence |
|---|---|---|
| **Mistral 7B** (Instruct), the LLM tested as System 2 | Mistral AI | Apache 2.0 ([announcement](https://mistral.ai/news/announcing-mistral-7b)) |
| **Ollama**, which runs Mistral on this computer | Ollama | MIT ([github.com/ollama/ollama](https://github.com/ollama/ollama)) |

The LLM's answers (`llm_scores.csv`, the LLM plans in `strategy_plans.csv`) are
outputs of Mistral 7B. The Apache 2.0 licence places no limits on them.

Cite the model as: Jiang, A. Q., Sablayrolles, A., Mensch, A., et al. (2023).
*Mistral 7B.* arXiv:2310.06825.

---

## 3. Python libraries

All the libraries this project imports are open source under permissive licences
(MIT, BSD or Apache), which allow use in a public project as long as their
licence notices are kept. `requirements.txt` lists them with exact versions.

| Library | Used for | Licence | Cite as |
|---|---|---|---|
| XGBoost | System 1 model | Apache 2.0 | Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794. https://doi.org/10.1145/2939672.2939785 |
| SHAP | System 1 explanations | MIT | Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems 30*. Lundberg, S. M., et al. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2, 56–67. https://doi.org/10.1038/s42256-019-0138-9 |
| scikit-learn | Train/test splits, error measures | BSD-3-Clause | Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830. |
| pandas | Tables of data | BSD-3-Clause | McKinney, W. (2010). Data structures for statistical computing in Python. *Proceedings of the 9th Python in Science Conference*, 56–61. |
| NumPy | Numbers and arrays | BSD-3-Clause | Harris, C. R., et al. (2020). Array programming with NumPy. *Nature*, 585, 357–362. https://doi.org/10.1038/s41586-020-2649-2 |
| SciPy | Spearman correlation | BSD-3-Clause | Virtanen, P., et al. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods*, 17, 261–272. https://doi.org/10.1038/s41592-019-0686-2 |
| Matplotlib | Charts | Matplotlib licence (PSF-based, permissive) | Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering*, 9(3), 90–95. https://doi.org/10.1109/MCSE.2007.55 |
| seaborn | Heatmap | BSD-3-Clause | Waskom, M. L. (2021). seaborn: Statistical data visualization. *Journal of Open Source Software*, 6(60), 3021. https://doi.org/10.21105/joss.03021 |
| Gradio | Dashboard | Apache 2.0 | Abid, A., et al. (2019). Gradio: Hassle-free sharing and testing of ML models in the wild. arXiv:1906.02569. |
| Plotly | Dashboard charts | MIT | Plotly Technologies Inc. (2015). *Collaborative data science.* Montréal, QC. https://plot.ly |
| ollama (Python) | Talking to Ollama | MIT | — |
| Requests | Downloading data | Apache 2.0 | — |
| Python | The programming language | Python Software Foundation License | Python Software Foundation. *Python Language Reference*, version 3.12. https://www.python.org |

These libraries install a few others of their own. Some of those (certifi, tqdm,
orjson) are under the Mozilla Public License 2.0, which only matters if you
change and share their own files. This project does neither.

---

## 4. Methods from published research

- **Conformal prediction** (the 95% range in `save_results.py`): Vovk, V.,
  Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World.*
  Springer. Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., & Wasserman, L.
  (2018). Distribution-free predictive inference for regression. *Journal of the
  American Statistical Association*, 113(523), 1094–1111.
- **Quantile regression** (quantile XGBoost bounds): Koenker, R., & Bassett, G.
  (1978). Regression quantiles. *Econometrica*, 46(1), 33–50.
- **2% inflation as the lowest goal** (strategy planner): the inflation target
  used by many central banks, e.g. the European Central Bank and the US Federal
  Reserve, and the midpoint of the Reserve Bank of India's 2–6% band. It is a
  widely used convention, not a single source.
- **Spearman rank correlation**: Spearman, C. (1904). The proof and measurement
  of association between two things. *American Journal of Psychology*, 15(1),
  72–101.
- **Shapley values** (the idea behind SHAP): Shapley, L. S. (1953). A value for
  n-person games. In *Contributions to the Theory of Games II*, 307–317.
  Princeton University Press.

---

## 5. Numbers used in the environmental estimate

`environmental_cost.py` uses these published figures:

- **Processor power, 15 W** (12.5–25 W configurable): Intel Core i7-10610U
  specifications, [Intel](https://www.intel.com/content/www/us/en/products/sku/201896/intel-core-i710610u-processor-8m-cache-up-to-4-90-ghz/specifications.html)
  and [Notebookcheck](https://www.notebookcheck.net/i7-10610U-vs-i7-10810U_12200_11935.247596.0.html).
- **CO2, 0.727 kg per kWh**: Central Electricity Authority, Government of India
  (2024). *CO2 Baseline Database for the Indian Power Sector, Version 20.0*,
  weighted average emission rate for FY 2023-24. https://cea.nic.in/
- **Water, 3.5 litres per kWh**: Ministry of Environment, Forest and Climate
  Change, Government of India (2015). Environment (Protection) Amendment Rules,
  2015, the limit on water use by thermal power plants (3.5 m³/MWh). Reported in
  [Down To Earth](https://www.downtoearth.org.in/news/widespread-non-compliance-as-thermal-power-plants-don-t-meet-water-use-norms-65391).

---

## 6. Acknowledgements

- Project author: **Jay Prafulkumar Joshi** (enrolment number 26210062), IIT Gandhinagar, for the course ES678.
- The code follows the **NationPulse Python Coding Guide**, written for this
  project (each script's header names the part of the guide it comes from).
- Code written with the help of **Claude Code** (Anthropic), an AI coding assistant.
