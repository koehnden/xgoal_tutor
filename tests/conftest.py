"""Testing configuration providing lightweight stubs for external dependencies."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional


# --------------------------------------------------------------------------------------
# Minimal FastAPI stub
# --------------------------------------------------------------------------------------
if "fastapi" not in sys.modules:  # pragma: no cover - import side effect
    fastapi_module = ModuleType("fastapi")

    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: Any) -> None:
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class _Route:
        def __init__(self, method: str, path: str, endpoint: Callable[..., Any]) -> None:
            self.method = method
            self.path = path
            self.endpoint = endpoint

    class FastAPI:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.routes: List[_Route] = []

        def post(
            self, path: str, response_model: Optional[Any] = None
        ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
            return self._add_route("POST", path)

        def get(
            self, path: str, response_model: Optional[Any] = None
        ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
            return self._add_route("GET", path)

        def _add_route(
            self, method: str, path: str
        ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
            def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
                self.routes.append(_Route(method, path, func))
                return func

            return decorator

    fastapi_module.FastAPI = FastAPI
    fastapi_module.HTTPException = HTTPException
    sys.modules["fastapi"] = fastapi_module


# --------------------------------------------------------------------------------------
# Minimal Pydantic stub
# --------------------------------------------------------------------------------------
if "pydantic" not in sys.modules:  # pragma: no cover - import side effect
    pydantic_module = ModuleType("pydantic")

    class _UnsetType:
        pass

    _UNSET = _UnsetType()

    @dataclass
    class FieldInfo:
        default: Any = _UNSET
        description: Optional[str] = None

    def Field(default: Any = _UNSET, **kwargs: Any) -> FieldInfo:
        return FieldInfo(default=default, description=kwargs.get("description"))

    def field_validator(*field_names: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            target = func.__func__ if isinstance(func, classmethod) else func
            existing = getattr(target, "__validator_fields__", ())
            setattr(target, "__validator_fields__", existing + field_names)
            return func

        return decorator

    class BaseModel:
        def __init_subclass__(cls, **kwargs: Any) -> None:
            super().__init_subclass__(**kwargs)
            cls.__field_defaults__ = {}
            cls.__required_fields__ = set()
            cls.__field_validators__ = {}

            for base in cls.__mro__[1:]:
                cls.__field_defaults__.update(getattr(base, "__field_defaults__", {}))
                cls.__required_fields__.update(getattr(base, "__required_fields__", set()))
                for field, validators in getattr(base, "__field_validators__", {}).items():
                    cls.__field_validators__.setdefault(field, []).extend(list(validators))

            annotations = getattr(cls, "__annotations__", {})
            for name in annotations:
                value = cls.__dict__.get(name, _UNSET)
                if isinstance(value, FieldInfo):
                    if value.default is _UNSET:
                        cls.__required_fields__.add(name)
                        default_value = None
                    else:
                        default_value = value.default
                    setattr(cls, name, default_value)
                elif value is _UNSET:
                    cls.__required_fields__.add(name)
                    default_value = None
                else:
                    default_value = value
                cls.__field_defaults__[name] = default_value

            for attr_name, attr_value in cls.__dict__.items():
                candidate = attr_value.__func__ if isinstance(attr_value, classmethod) else attr_value
                fields = getattr(candidate, "__validator_fields__", ())
                if not fields:
                    continue
                callable_obj = (
                    attr_value.__get__(cls, cls)
                    if isinstance(attr_value, (classmethod, staticmethod))
                    else attr_value
                )
                for field in fields:
                    cls.__field_validators__.setdefault(field, []).append(callable_obj)

        def __init__(self, **data: Any) -> None:
            values: Dict[str, Any] = {}
            for name, default in self.__field_defaults__.items():
                if name in data:
                    value = data.pop(name)
                else:
                    if name in self.__required_fields__:
                        raise ValueError(f"Field '{name}' is required")
                    value = default
                validators = self.__field_validators__.get(name, [])
                for validator in validators:
                    value = validator(value)
                values[name] = value

            if data:
                unexpected = ", ".join(sorted(data.keys()))
                raise ValueError(f"Unexpected fields: {unexpected}")

            for name, value in values.items():
                setattr(self, name, value)

        def model_dump(self) -> Dict[str, Any]:
            return dict(self.__dict__)

        def __repr__(self) -> str:  # pragma: no cover - debug helper
            fields = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
            return f"{self.__class__.__name__}({fields})"

    pydantic_module.BaseModel = BaseModel
    pydantic_module.Field = Field
    pydantic_module.field_validator = field_validator
    sys.modules["pydantic"] = pydantic_module
