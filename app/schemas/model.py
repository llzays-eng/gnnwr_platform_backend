"""建模契约：对齐独立前端 TrainRequest / ModelTask / ModelResult / BaselineComparison。"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DatasetSplit(BaseModel):
    train: float = 0.7
    val: float = 0.1
    test: float = 0.2


class ModelHyperparams(BaseModel):
    """同时接受前端字段（hidden_layers / max_epochs / learning_rate）与主仓别名。"""

    hidden_layers: list[int] = Field(default_factory=lambda: [64, 32])
    hidden: list[int] | None = None
    activation: str = "tanh"
    dropout: float = 0.0
    batch_size: int | None = None
    max_epochs: int = 200
    max_epoch: int | None = None
    learning_rate: float = 3e-3
    lr: float | None = None
    early_stopping_patience: int = 30
    patience: int | None = None
    split: DatasetSplit = Field(default_factory=DatasetSplit)
    random_seed: int = 42
    l2_coef: float = 1e-3

    def normalized(self) -> dict:
        hidden = self.hidden or self.hidden_layers
        max_epoch = self.max_epoch or self.max_epochs
        lr = self.lr if self.lr is not None else self.learning_rate
        patience = self.patience if self.patience is not None else self.early_stopping_patience
        return {
            "hidden": list(hidden),
            "dropout": self.dropout,
            "lr": lr,
            "max_epoch": int(max_epoch),
            "patience": int(patience),
            "batch_size": self.batch_size,
            "l2_coef": self.l2_coef,
            "test_ratio": self.split.test,
            "valid_ratio": self.split.val,
            "seed": self.random_seed,
            "activation": self.activation,
            # 原样保留前端形状，方便 GET 任务回显
            "hidden_layers": list(hidden),
            "max_epochs": int(max_epoch),
            "learning_rate": lr,
            "early_stopping_patience": int(patience),
            "split": self.split.model_dump(),
            "random_seed": self.random_seed,
        }


class TrainRequest(BaseModel):
    project_id: str
    dataset_id: str
    model_type: str = Field(default="GNNWR", description="GNNWR | GTNNWR")
    y_column: str
    x_columns: list[str]
    spatial_columns: list[str]
    temporal_column: str | None = None
    test_ratio: float | None = None
    valid_ratio: float | None = None
    hyperparams: ModelHyperparams = Field(default_factory=ModelHyperparams)

    @field_validator("model_type")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()

    @model_validator(mode="after")
    def _check(self):
        if self.y_column in self.x_columns:
            raise ValueError(f"目标变量 {self.y_column} 不能同时作为自变量")
        if self.model_type == "GTNNWR" and not self.temporal_column:
            raise ValueError("GTNNWR 为时空模式，必须指定时间列")
        if self.model_type == "GNNWR" and self.temporal_column:
            raise ValueError("GNNWR 为纯空间模式，不接受时间列")
        if len(self.spatial_columns) < 2:
            raise ValueError("spatial_columns 必须为 [经度列, 纬度列]")
        return self


class TaskProgress(BaseModel):
    epoch: int = 0
    total_epochs: int = 0
    train_loss: float | None = None
    val_loss: float | None = None
    elapsed_s: float = 0.0
    eta_s: float | None = None


class TaskError(BaseModel):
    code: str
    message: str
    retryable: bool = False


class ModelTaskOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    project_id: str
    dataset_id: str
    model_type: str
    y_column: str
    x_columns: list[str]
    spatial_columns: list[str]
    temporal_column: str | None = None
    hyperparams: dict = Field(default_factory=dict)
    status: str
    progress: TaskProgress | None = None
    error: TaskError | None = None
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    # 主仓兼容
    task_id: str | None = None


class CoefficientSummary(BaseModel):
    variable: str
    mean: float
    std: float
    min: float
    q05: float
    q25: float
    q50: float
    q75: float
    q95: float
    max: float
    positive_ratio: float


class ResidualSummary(BaseModel):
    mean: float
    std: float
    min: float
    max: float
    histogram: list[list[float]] = Field(default_factory=list)
    morans_i: float | None = None


class ModelResultOut(BaseModel):
    task_id: str
    r2: float | None = None
    rmse: float | None = None
    mae: float | None = None
    aicc: float | None = None
    coefficients_summary: list[dict] = Field(default_factory=list)
    residual_summary: dict = Field(default_factory=dict)
    sample_count: int = 0
    loss_history: list[list[float]] = Field(default_factory=list)
    # 主仓兼容
    status: str | None = None
    model_type: str | None = None
    metrics: dict | None = None
    beta_ols: dict = Field(default_factory=dict)
    coefficients_key: str | None = None


class BaselineEntry(BaseModel):
    model: str
    is_target: bool = False
    r2: float | None = None
    rmse: float | None = None
    mae: float | None = None
    aicc: float | None = None
    coefficients_summary: list[dict] | None = None


class BaselineComparisonOut(BaseModel):
    task_id: str
    entries: list[BaselineEntry]
    best_by_metric: dict[str, str]
    model_type: str | None = None
    main: dict | None = None
    baselines: list[dict] | None = None


class CompareOut(BaselineComparisonOut):
    """别名。"""

    model_type: str | None = None
    main: dict | None = None
    baselines: list[dict] | None = None


class GeoJsonPoint(BaseModel):
    type: str = "Point"
    coordinates: list[float]


class CoefficientPoint(BaseModel):
    feature_id: str
    geom: GeoJsonPoint
    coefficients: dict[str, float]
    local_r2: float | None = None
    residual: float | None = None
    observed: float | None = None
    predicted: float | None = None
    time: float | None = None


class CoefficientPage(BaseModel):
    points: list[CoefficientPoint]
    next_cursor: str | None = None
    total: int | None = None
    crs: str = "WGS84"
    truncated: bool = False
    # 主仓兼容整包
    columns: list[str] | None = None
    temporal: bool | None = None
    times: list[Any] | None = None


class TaskLogsOut(BaseModel):
    lines: list[dict]


class CancelOut(BaseModel):
    task_id: str
    status: str
    revoked: bool = False
