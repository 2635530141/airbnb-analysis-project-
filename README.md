# Airbnb Analysis Project

MaCSS

用于 Airbnb 数据清理、探索性分析与可视化的项目。

## 环境准备

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

启动 Jupyter：

```bash
jupyter notebook
```

## 项目结构

- `data/raw/`：原始数据，不纳入 Git。
- `data/processed/`：清洗和转换后的数据。
- `notebooks/`：探索、分析和可视化 notebooks。
- `src/`：可复用的分析代码。
- `outputs/`：导出的图表和分析结果。

将数据文件放入 `data/raw/`，并从 `notebooks/` 中开始分析。
