from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil
from typing import Any, Generic, TypeVar

from pydantic import BaseModel
from sqlalchemy import ColumnElement, delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm.interfaces import LoaderOption

ModelType = TypeVar("ModelType", bound=DeclarativeBase)
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)
T = TypeVar("T")


@dataclass
class PaginatedResult(Generic[T]):
    """
    Container for a single page of query results.

    :param total: The total number of rows matching the query (across all pages).
    :type total: int
    :param items: The rows belonging to the current page.
    :type items: List[T]
    :param page: The 1-based page number.
    :type page: int
    :param per_page: The number of items requested per page.
    :type per_page: int
    :param num_pages: The total number of pages (at least 1).
    :type num_pages: int
    """

    total: int
    items: list[T]
    page: int
    per_page: int
    num_pages: int


class BaseManager(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    """
    Generic async CRUD manager — adapted from pissync-core BaseManager.

    ponytail: actor scoping, lifecycle events, and author map removed — not needed here.
    Add them back when multi-tenancy or event streams are required.

    :param ModelType: The SQLAlchemy model the manager operates on.
    :param CreateSchemaType: The Pydantic schema used for creation.
    :param UpdateSchemaType: The Pydantic schema used for updates.
    """

    def __init__(self, model: type[ModelType]) -> None:
        """
        Bind the manager to a concrete model class.

        :param model: The SQLAlchemy model class to manage.
        :type model: Type[ModelType]
        """
        self.model = model

    def _order_columns(self, order_by: str | None) -> list[ColumnElement]:
        """
        Parse an order string into ordered column expressions.

        Accepts ``"field"`` (ascending), ``"-field"`` (descending) and
        comma-separated combinations such as ``"a,-b"``. Unknown field names are
        silently ignored.

        :param order_by: The order specification, or None for no ordering.
        :type order_by: Optional[str]
        :return: The resolved column expressions in order.
        :rtype: List[ColumnElement]
        """
        columns: list[ColumnElement] = []
        if not order_by:
            return columns
        for token in order_by.split(","):
            token = token.strip()
            if not token:
                continue
            desc = token.startswith("-")
            name = token[1:] if desc else token
            if hasattr(self.model, name):
                col = getattr(self.model, name)
                columns.append(col.desc() if desc else col.asc())

        return columns

    async def create(
        self,
        db: AsyncSession,
        obj_in: CreateSchemaType | dict[str, Any],
        auto_commit: bool = True,
    ) -> ModelType:
        """
        Persist a new instance from a schema or a plain dict.

        :param db: The active async session.
        :type db: AsyncSession
        :param obj_in: The data to create the instance from.
        :type obj_in: Union[CreateSchemaType, Dict[str, Any]]
        :param auto_commit: Whether to commit immediately. Pass False to batch
            multiple creates within an outer transaction.
        :type auto_commit: bool
        :return: The newly created and refreshed instance.
        :rtype: ModelType
        """
        data = obj_in if isinstance(obj_in, dict) else obj_in.model_dump(exclude_unset=True)
        db_obj = self.model(**data)
        db.add(db_obj)
        await db.flush()
        if auto_commit:
            await db.commit()
        await db.refresh(db_obj)

        return db_obj

    async def get(
        self,
        db: AsyncSession,
        id: Any,
        options: Sequence[LoaderOption] | None = None,
    ) -> ModelType | None:
        """
        Fetch a single instance by primary key.

        :param db: The active async session.
        :type db: AsyncSession
        :param id: The primary key value.
        :type id: Any
        :param options: Optional loader options (e.g. eager loading).
        :type options: Optional[Sequence[LoaderOption]]
        :return: The matching instance, or None if not found.
        :rtype: Optional[ModelType]
        """
        query = select(self.model).where(self.model.id == id)
        if options:
            query = query.options(*options)

        return (await db.execute(query)).scalar_one_or_none()

    async def get_by_field(
        self,
        db: AsyncSession,
        field_name: str,
        field_value: Any,
        options: Sequence[LoaderOption] | None = None,
    ) -> ModelType | None:
        """
        Fetch a single instance matching an arbitrary field.

        :param db: The active async session.
        :type db: AsyncSession
        :param field_name: The model attribute to filter on.
        :type field_name: str
        :param field_value: The value the field must equal.
        :type field_value: Any
        :param options: Optional loader options (e.g. eager loading).
        :type options: Optional[Sequence[LoaderOption]]
        :raises AttributeError: If the model has no such field.
        :return: The matching instance, or None if not found.
        :rtype: Optional[ModelType]
        """
        if not hasattr(self.model, field_name):
            raise AttributeError(f"{self.model.__name__} has no field '{field_name}'")
        field = getattr(self.model, field_name)
        query = select(self.model).where(field == field_value)
        if options:
            query = query.options(*options)
        return (await db.execute(query)).scalar_one_or_none()

    async def get_multi(
        self,
        db: AsyncSession,
        skip: int = 0,
        limit: int = 100,
        filters: dict[str, Any] | None = None,
        order_by: str | None = None,
        expressions: Sequence[ColumnElement] | None = None,
        options: Sequence[LoaderOption] | None = None,
    ) -> list[ModelType]:
        """
        Fetch multiple instances with optional filtering, ordering and paging.

        A filter value that is a list is matched with ``IN``; any other value is
        matched with equality. Unknown filter keys are ignored.

        :param db: The active async session.
        :type db: AsyncSession
        :param skip: The number of rows to skip (offset).
        :type skip: int
        :param limit: The maximum number of rows to return.
        :type limit: int
        :param filters: A mapping of field name to value (or list of values).
        :type filters: Optional[Dict[str, Any]]
        :param order_by: An order specification (see :meth:`_order_columns`).
        :type order_by: Optional[str]
        :param expressions: Extra SQLAlchemy filter expressions.
        :type expressions: Optional[Sequence[ColumnElement]]
        :param options: Optional loader options (e.g. eager loading).
        :type options: Optional[Sequence[LoaderOption]]
        :return: The matching instances.
        :rtype: List[ModelType]
        """
        query = select(self.model)
        if expressions:
            for expr in expressions:
                query = query.where(expr)
        if filters:
            for name, value in filters.items():
                if hasattr(self.model, name):
                    col = getattr(self.model, name)
                    query = query.where(col.in_(value) if isinstance(value, list) else col == value)
        for col in self._order_columns(order_by):
            query = query.order_by(col)
        if options:
            query = query.options(*options)
        return list((await db.execute(query.offset(skip).limit(limit))).scalars().all())

    async def count(
        self,
        db: AsyncSession,
        filters: dict[str, Any] | None = None,
        expressions: Sequence[ColumnElement] | None = None,
    ) -> int:
        """
        Count rows matching the given filters and expressions.

        :param db: The active async session.
        :type db: AsyncSession
        :param filters: A mapping of field name to value (or list of values).
        :type filters: Optional[Dict[str, Any]]
        :param expressions: Extra SQLAlchemy filter expressions.
        :type expressions: Optional[Sequence[ColumnElement]]
        :return: The number of matching rows.
        :rtype: int
        """
        query = select(func.count(self.model.id))
        if expressions:
            for expr in expressions:
                query = query.where(expr)
        if filters:
            for name, value in filters.items():
                if hasattr(self.model, name):
                    col = getattr(self.model, name)
                    query = query.where(col.in_(value) if isinstance(value, list) else col == value)
        return (await db.execute(query)).scalar()

    async def paginate(
        self,
        db: AsyncSession,
        page: int = 1,
        per_page: int = 20,
        filters: dict[str, Any] | None = None,
        order_by: str | None = None,
        expressions: Sequence[ColumnElement] | None = None,
        options: Sequence[LoaderOption] | None = None,
    ) -> "PaginatedResult[ModelType]":
        """
        Fetch a single page of results together with pagination metadata.

        :param db: The active async session.
        :type db: AsyncSession
        :param page: The 1-based page number.
        :type page: int
        :param per_page: The number of items per page.
        :type per_page: int
        :param filters: A mapping of field name to value (or list of values).
        :type filters: Optional[Dict[str, Any]]
        :param order_by: An order specification (see :meth:`_order_columns`).
        :type order_by: Optional[str]
        :param expressions: Extra SQLAlchemy filter expressions.
        :type expressions: Optional[Sequence[ColumnElement]]
        :param options: Optional loader options (e.g. eager loading).
        :type options: Optional[Sequence[LoaderOption]]
        :return: The page items plus total/page/per_page/num_pages.
        :rtype: PaginatedResult[ModelType]
        """
        total = await self.count(db, filters=filters, expressions=expressions)
        items = await self.get_multi(
            db,
            skip=(page - 1) * per_page,
            limit=per_page,
            filters=filters,
            order_by=order_by,
            expressions=expressions,
            options=options,
        )
        return PaginatedResult(
            total=total,
            items=items,
            page=page,
            per_page=per_page,
            num_pages=max(1, ceil(total / per_page)),
        )

    async def update(
        self,
        db: AsyncSession,
        id: Any,
        obj_in: UpdateSchemaType | dict[str, Any],
    ) -> ModelType | None:
        """
        Update an instance located by primary key.

        :param db: The active async session.
        :type db: AsyncSession
        :param id: The primary key of the instance to update.
        :type id: Any
        :param obj_in: The fields to update.
        :type obj_in: Union[UpdateSchemaType, Dict[str, Any]]
        :return: The updated instance, or None if it does not exist.
        :rtype: Optional[ModelType]
        """
        db_obj = await self.get(db, id)
        if not db_obj:
            return None
        return await self.update_instance(db, db_obj, obj_in)

    async def update_instance(
        self,
        db: AsyncSession,
        db_obj: ModelType,
        obj_in: UpdateSchemaType | dict[str, Any],
    ) -> ModelType:
        """
        Apply field changes to an already-loaded instance and persist them.

        Only attributes that exist on the instance are set; unknown keys are
        ignored.

        :param db: The active async session.
        :type db: AsyncSession
        :param db_obj: The instance to mutate.
        :type db_obj: ModelType
        :param obj_in: The fields to update.
        :type obj_in: Union[UpdateSchemaType, Dict[str, Any]]
        :return: The updated and refreshed instance.
        :rtype: ModelType
        """
        data = obj_in if isinstance(obj_in, dict) else obj_in.model_dump(exclude_unset=True)
        for field, value in data.items():
            if hasattr(db_obj, field):
                setattr(db_obj, field, value)
        await db.flush()
        await db.commit()
        await db.refresh(db_obj)
        return db_obj

    async def update_bulk(
        self,
        db: AsyncSession,
        filters: dict[str, Any],
        obj_in: UpdateSchemaType | dict[str, Any],
    ) -> int:
        """
        Update every row matching the filters in a single statement.

        :param db: The active async session.
        :type db: AsyncSession
        :param filters: A mapping of field name to value to match.
        :type filters: Dict[str, Any]
        :param obj_in: The fields to set on the matching rows.
        :type obj_in: Union[UpdateSchemaType, Dict[str, Any]]
        :return: The number of rows updated.
        :rtype: int
        """
        data = obj_in if isinstance(obj_in, dict) else obj_in.model_dump(exclude_unset=True)
        query = update(self.model)
        for name, value in filters.items():
            if hasattr(self.model, name):
                query = query.where(getattr(self.model, name) == value)
        result = await db.execute(query.values(**data))
        await db.commit()
        return result.rowcount

    async def delete(self, db: AsyncSession, id: Any) -> bool:
        """
        Delete an instance located by primary key.

        :param db: The active async session.
        :type db: AsyncSession
        :param id: The primary key of the instance to delete.
        :type id: Any
        :return: True if a row was deleted, False if none matched.
        :rtype: bool
        """
        db_obj = await self.get(db, id)
        if not db_obj:
            return False
        await db.delete(db_obj)
        await db.commit()
        return True

    async def delete_by_field(self, db: AsyncSession, field_name: str, field_value: Any) -> int:
        """
        Delete every row whose field equals the given value.

        :param db: The active async session.
        :type db: AsyncSession
        :param field_name: The model attribute to filter on.
        :type field_name: str
        :param field_value: The value the field must equal.
        :type field_value: Any
        :raises AttributeError: If the model has no such field.
        :return: The number of rows deleted.
        :rtype: int
        """
        if not hasattr(self.model, field_name):
            raise AttributeError(f"{self.model.__name__} has no field '{field_name}'")
        result = await db.execute(delete(self.model).where(getattr(self.model, field_name) == field_value))
        await db.commit()
        return result.rowcount

    async def delete_bulk(self, db: AsyncSession, filters: dict[str, Any]) -> int:
        """
        Delete every row matching the filters in a single statement.

        :param db: The active async session.
        :type db: AsyncSession
        :param filters: A mapping of field name to value to match.
        :type filters: Dict[str, Any]
        :return: The number of rows deleted.
        :rtype: int
        """
        query = delete(self.model)
        for name, value in filters.items():
            if hasattr(self.model, name):
                query = query.where(getattr(self.model, name) == value)
        result = await db.execute(query)
        await db.commit()
        return result.rowcount

    async def exists(self, db: AsyncSession, id: Any) -> bool:
        """
        Check whether an instance with the given primary key exists.

        :param db: The active async session.
        :type db: AsyncSession
        :param id: The primary key value.
        :type id: Any
        :return: True if a matching row exists.
        :rtype: bool
        """
        result = await db.execute(select(self.model.id).where(self.model.id == id))
        return result.scalar_one_or_none() is not None

    async def exists_by_field(self, db: AsyncSession, field_name: str, field_value: Any) -> bool:
        """
        Check whether a row exists whose field equals the given value.

        :param db: The active async session.
        :type db: AsyncSession
        :param field_name: The model attribute to filter on.
        :type field_name: str
        :param field_value: The value the field must equal.
        :type field_value: Any
        :raises AttributeError: If the model has no such field.
        :return: True if a matching row exists.
        :rtype: bool
        """
        if not hasattr(self.model, field_name):
            raise AttributeError(f"{self.model.__name__} has no field '{field_name}'")
        result = await db.execute(select(self.model.id).where(getattr(self.model, field_name) == field_value))
        return result.scalar_one_or_none() is not None
