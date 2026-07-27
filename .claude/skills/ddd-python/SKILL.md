---
name: ddd-python
description: Tactical Domain-Driven Design patterns for Python projects. Uses Pydantic (or dataclasses when Pydantic isn't available) for value objects and entities, ABCs for ports, and SQLAlchemy for repository implementations. Use when structuring a Python service with complex business rules, implementing entities/value objects/aggregates, defining repository interfaces and SQLAlchemy implementations, adding domain events or application service layers, refactoring Django/FastAPI code towards hexagonal architecture, deciding how to model a domain concept as an entity vs. a value object, or wiring DDD patterns into a FastAPI or Django project without leaking domain logic into views.
---

# DDD Python

Tactical Domain-Driven Design patterns for Python projects. Uses Pydantic and if is not possible dataclasses, for value objects and entities, ABCs for ports, and SQLAlchemy for the repository implementation.

## When to Activate

- Structuring a Python service with complex business rules
- Implementing entities, value objects, or aggregates in Python
- Defining repository interfaces and SQLAlchemy implementations
- Adding domain events and application service layers
- Refactoring Django/FastAPI code towards hexagonal architecture
- Deciding how to model a domain concept as an entity vs. a value object in Python
- Wiring DDD patterns into a FastAPI or Django project without leaking domain logic into views

---

## Entities with Pydantic V2

Entities have an identity that persists as their state changes. Two entities are equal when they have the same type and id, regardless of their other field values.Use immutable Pydantic models for entities. Business operations return a new instance with the same identity.

```python

from datetime import UTC, datetime
from typing import Self
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, field_validator


class User(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    id: UUID
    email: str
    name: str
    created_at: datetime

    @field_validator("email", mode="before")
    @classmethod
    def normalize_email(cls, value: object) -> str:
        email = str(value).strip().lower()
        if not email:
            raise ValueError("Email must not be blank")
        return email

    @field_validator("name", mode="before")
    @classmethod
    def normalize_name(cls, value: object) -> str:
        name = str(value).strip()
        if not name:
            raise ValueError("Name must not be blank")
        return name

    @classmethod
    def create(cls, *, email: str, name: str) -> Self:
        return cls(
            id=uuid4(),
            email=email,
            name=name,
            created_at=datetime.now(UTC),
        )

    def rename(self, new_name: str) -> Self:
        # model_copy(update=...) does not run Pydantic validation, so validate
        # the changed value explicitly before creating the new entity state.
        normalized_name = self.__class__.model_fields["name"].annotation
        del normalized_name  # validation is performed by constructing below

        data = self.model_dump()
        data["name"] = new_name
        return self.__class__.model_validate(data)

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash((type(self), self.id))



```

Important rules:
- Use ConfigDict(frozen=True) so entity state cannot be mutated accidentally.
- Keep id required. Generate it in create(); repositories must provide the persisted identity when rebuilding an entity.
- Use datetime.now(UTC), producing a timezone-aware timestamp.
- Put invariants in Pydantic validators so they apply both to newly created and reconstituted entities.
- Do not rely on Pydantic's default equality, which compares model values.
- Avoid model_copy(update=...) for untrusted changes because updates are not validated.

---

---

## Entities with dataclasses

Entities have identity (an `id` field) that persists over time. Equality is by identity, not value.

```python
from dataclasses import dataclass, field
from uuid import UUID, uuid4
from datetime import datetime


@dataclass
class User:
    id: UUID
    email: str
    name: str
    created_at: datetime

    @classmethod
    def create(cls, email: str, name: str) -> "User":
        return cls(
            id=uuid4(),
            email=email.lower().strip(),
            name=name.strip(),
            created_at=datetime.utcnow(),
        )

    def rename(self, new_name: str) -> "User":
        if not new_name.strip():
            raise ValueError("Name must not be blank")
        from dataclasses import replace
        return replace(self, name=new_name.strip())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, User):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)
```

---

## Value Objects with dataclasses

Value objects have no identity — equality is by value. Always immutable.

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Money:
    amount: int   # store in minor units (cents)
    currency: str

    def __post_init__(self) -> None:
        if self.amount < 0:
            raise ValueError("Amount must not be negative")
        if len(self.currency) != 3:
            raise ValueError("Currency must be a 3-letter ISO code")

    def add(self, other: "Money") -> "Money":
        if self.currency != other.currency:
            raise ValueError("Cannot add different currencies")
        return Money(self.amount + other.amount, self.currency)

    def __str__(self) -> str:
        return f"{self.amount / 100:.2f} {self.currency}"


@dataclass(frozen=True)
class EmailAddress:
    value: str

    def __post_init__(self) -> None:
        if "@" not in self.value:
            raise ValueError(f"Invalid email: {self.value!r}")
        object.__setattr__(self, "value", self.value.lower().strip())
```

---

## Value Objects with pydantic V2

Value objects have no identity. Equality and hashing are based on their field values. They should be immutable and enforce their invariants when constructed.
With Pydantic v2, use frozen=True. Field validators replace dataclass __post_init__.


```python
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ValueObject(BaseModel):
    """Base class for immutable domain value objects."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
    )


class Money(ValueObject):
    amount: int = Field(ge=0)  # Stored in minor units (cents).
    currency: str = Field(
        min_length=3,
        max_length=3,
        pattern=r"^[A-Z]{3}$",
    )

    @field_validator("currency", mode="before")
    @classmethod
    def normalize_currency(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().upper()
        return value

    def add(self, other: Self) -> Self:
        if self.currency != other.currency:
            raise ValueError("Cannot add different currencies")

        return type(self)(
            amount=self.amount + other.amount,
            currency=self.currency,
        )

    def __str__(self) -> str:
        return f"{self.amount / 100:.2f} {self.currency}"


class EmailAddress(ValueObject):
    value: str

    @field_validator("value", mode="before")
    @classmethod
    def normalize_and_validate(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("Email address must be a string")

        normalized = value.strip().lower()

        if "@" not in normalized:
            raise ValueError(f"Invalid email: {value!r}")

        return normalized

    def __str__(self) -> str:
        return self.value
```

Pydantic already supplies the desired value-object behavior:
- frozen=True prevents attribute reassignment.
- Equality compares models by their field values.
- Frozen models are hashable when all their fields are hashable.
- strict=True prevents coercions such as "100" into 100.
- Operations construct a new validated value object instead of mutating the existing one.
- Validation failures are exposed to callers as Pydantic ValidationError.
- Avoid model_copy(update=...) for value-object operations because Pydantic does not validate update values. Constructing a new instance, as Money.add() does, ensures all invariants run again.

---

---

## Aggregates with dataclasses

Aggregates group entities and value objects under a single root. External code only interacts with the root. Invariants are enforced inside the aggregate.

```python
from dataclasses import dataclass, field
from uuid import UUID, uuid4
from typing import List


@dataclass
class OrderLine:
    product_id: UUID
    quantity: int
    unit_price: Money


@dataclass
class Order:
    id: UUID
    customer_id: UUID
    lines: List[OrderLine] = field(default_factory=list)
    _events: List["DomainEvent"] = field(default_factory=list, repr=False, compare=False)

    @classmethod
    def create(cls, customer_id: UUID) -> "Order":
        order = cls(id=uuid4(), customer_id=customer_id)
        order._events.append(OrderCreated(order_id=order.id, customer_id=customer_id))
        return order

    def add_line(self, product_id: UUID, quantity: int, unit_price: Money) -> None:
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
        self.lines.append(OrderLine(product_id, quantity, unit_price))

    @property
    def total(self) -> Money:
        if not self.lines:
            return Money(0, "USD")
        result = Money(0, self.lines[0].unit_price.currency)
        for line in self.lines:
            result = result.add(Money(line.quantity * line.unit_price.amount, line.unit_price.currency))
        return result

    def collect_events(self) -> List["DomainEvent"]:
        events, self._events = self._events, []
        return events
```

---

---

## Aggregates with Pydantic v2

Aggregates group entities and value objects under a single root. External code changes aggregate state through root methods so invariants remain centralized.
The aggregate root may be mutable, while its contained value objects and exposed collections remain immutable.

```python

from typing import Self
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

# Assumes Money, DomainEvent, and OrderCreated are defined elsewhere.


class OrderLine(BaseModel):
    """Immutable value object owned by the Order aggregate."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
    )

    product_id: UUID
    quantity: int = Field(gt=0)
    unit_price: Money


class Order(BaseModel):
    """Aggregate root for an order."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        validate_assignment=True,
    )

    id: UUID
    customer_id: UUID

    # A tuple prevents callers from mutating the collection with append().
    lines: tuple[OrderLine, ...] = ()

    # Private attributes are excluded from validation and serialization.
    _events: list["DomainEvent"] = PrivateAttr(default_factory=list)

    @classmethod
    def create(cls, *, customer_id: UUID) -> Self:
        order = cls(
            id=uuid4(),
            customer_id=customer_id,
        )
        order._events.append(
            OrderCreated(
                order_id=order.id,
                customer_id=customer_id,
            )
        )
        return order

    def add_line(
        self,
        *,
        product_id: UUID,
        quantity: int,
        unit_price: Money,
    ) -> None:
        if quantity <= 0:
            raise ValueError("Quantity must be positive")

        if self.lines:
            order_currency = self.lines[0].unit_price.currency
            if unit_price.currency != order_currency:
                raise ValueError("All order lines must use the same currency")

        line = OrderLine(
            product_id=product_id,
            quantity=quantity,
            unit_price=unit_price,
        )

        # Replacing the tuple triggers Pydantic assignment validation.
        self.lines = (*self.lines, line)

    @property
    def total(self) -> Money:
        if not self.lines:
            return Money(amount=0, currency="USD")

        currency = self.lines[0].unit_price.currency
        result = Money(amount=0, currency=currency)

        for line in self.lines:
            line_total = Money(
                amount=line.quantity * line.unit_price.amount,
                currency=line.unit_price.currency,
            )
            result = result.add(line_total)

        return result

    def collect_events(self) -> list["DomainEvent"]:
        events = list(self._events)
        self._events.clear()
        return events

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Order):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash((type(self), self.id))

```

---

## Domain Events with dataclasses

```python
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID


@dataclass(frozen=True)
class DomainEvent:
    occurred_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True)
class OrderCreated(DomainEvent):
    order_id: UUID
    customer_id: UUID


@dataclass(frozen=True)
class OrderShipped(DomainEvent):
    order_id: UUID
    tracking_number: str
```

---

---

## Domain Events with Pydantic v2

Domain events are immutable records describing something that already happened. Name them in the past tense and store a timezone-aware occurrence timestamp.

```python

from datetime import UTC, datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DomainEvent(BaseModel):
    """Base type for immutable domain events."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
    )

    occurred_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
    )


class OrderCreated(DomainEvent):
    order_id: UUID
    customer_id: UUID


class OrderShipped(DomainEvent):
    order_id: UUID
    tracking_number: str

    @field_validator("tracking_number")
    @classmethod
    def validate_tracking_number(cls, value: str) -> str:
        normalized = value.strip()

        if not normalized:
            raise ValueError("Tracking number must not be blank")

        return normalized

```

Key points:
- frozen=True prevents events from being changed after creation.
- default_factory creates a fresh timestamp for every event.
- datetime.now(UTC) is preferred over the deprecated, timezone-naive datetime.utcnow().
- extra="forbid" catches misspelled or unexpected event fields.
- strict=True avoids silent coercion in domain code.
- Subclasses inherit the timestamp and configuration.
- Pydantic models must be constructed with keyword arguments.
- Validators normalize and enforce event-specific invariants.

If events are sent through a message broker, persist an event envelope separately with fields such as event_id, event_type, aggregate_id, and schema version. Those are transport and delivery concerns rather than necessarily part of every domain event.

---

## Repository Interface (Port)

```python
from abc import ABC, abstractmethod
from uuid import UUID
from typing import Optional


class OrderRepository(ABC):
    @abstractmethod
    async def find_by_id(self, order_id: UUID) -> Optional[Order]:
        ...

    @abstractmethod
    async def save(self, order: Order) -> None:
        ...

    @abstractmethod
    async def delete(self, order_id: UUID) -> None:
        ...
```

---

## SQLAlchemy Repository (Adapter)

```python
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from uuid import UUID
from typing import Optional


class SqlAlchemyOrderRepository(OrderRepository):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def find_by_id(self, order_id: UUID) -> Optional[Order]:
        row = await self._session.get(OrderModel, order_id)
        return _to_domain(row) if row else None

    async def save(self, order: Order) -> None:
        model = _to_model(order)
        await self._session.merge(model)

    async def delete(self, order_id: UUID) -> None:
        row = await self._session.get(OrderModel, order_id)
        if row:
            await self._session.delete(row)


def _to_domain(row: "OrderModel") -> Order:
    return Order(id=row.id, customer_id=row.customer_id)


def _to_model(order: Order) -> "OrderModel":
    return OrderModel(id=order.id, customer_id=order.customer_id)
```

---

## Application Service

Application services orchestrate domain objects. They have no business logic — they coordinate reads, aggregate calls, event dispatch, and persistence.

```python
class CreateOrderUseCase:
    def __init__(
        self,
        order_repo: OrderRepository,
        event_bus: EventBus,
    ) -> None:
        self._orders = order_repo
        self._events = event_bus

    async def execute(self, customer_id: UUID) -> UUID:
        order = Order.create(customer_id)
        await self._orders.save(order)

        for event in order.collect_events():
            await self._events.publish(event)

        return order.id
```

---

## Key Rules

1. **Entities**: identity-based equality; always create via class method, not `__init__` directly
2. **Value objects**: `@dataclass(frozen=True)`; validate in `__post_init__`; return new instances for "mutations"
3. **Aggregates**: protect invariants inside; expose domain events via `collect_events()`
4. **Repositories**: define as ABC (port); implement with SQLAlchemy or any ORM (adapter)
5. **Application services**: no domain logic; orchestrate + commit + publish events
6. **Domain layer**: zero framework imports; no FastAPI, Django, or SQLAlchemy in domain models

## Related Skills

- `solid-principles` — Solid principles in python
- `code-smells` — A serie of code smells for python
- `dependency-injection-python` — Wiring DDD layers into FastAPI
- `design-patterns` — Wiring DDD layers into Django
- `python-oop` — PostgreSQL patterns for the repository layer
- `stop-using-none` — PostgreSQL patterns for the repository layer
