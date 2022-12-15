import logging
from pathlib import Path
from typing import List, Tuple


import pandas as pd
import pydantic
from pyam import IamDataFrame
from pyam.index import replace_index_labels
from pyam.logging import adjust_log_level
from pydantic.error_wrappers import ErrorWrapper

from nomenclature.codelist import CodeList, RegionCodeList, VariableCodeList
from nomenclature.error.region import RegionNotDefinedError
from nomenclature.processor import (
    RegionAggregationMapping,
    RegionProcessor,
    RequiredDataValidator,
)
from nomenclature.processor.requiredData import RequiredData
from nomenclature.processor.utils import get_relative_path
from nomenclature.validation import validate

logger = logging.getLogger(__name__)
SPECIAL_CODELIST = {"variable": VariableCodeList, "region": RegionCodeList}


class DataStructureDefinition:
    """Definition of datastructure codelists for dimensions used in the IAMC format"""

    def __init__(self, path, dimensions=["region", "variable"]):
        """

        Parameters
        ----------
        path : str or path-like
            The folder with the project definitions.
        dimensions : list of str, optional
            List of :meth:`CodeList` names. Each CodeList is initialized
            from a sub-folder of `path` of that name.
        """
        if not isinstance(path, Path):
            path = Path(path)

        if not path.is_dir():
            raise NotADirectoryError(f"Definitions directory not found: {path}")

        self.dimensions = dimensions
        for dim in self.dimensions:
            self.__setattr__(
                dim, SPECIAL_CODELIST.get(dim, CodeList).from_directory(dim, path / dim)
            )

        empty = [d for d in self.dimensions if not self.__getattribute__(d)]
        if empty:
            raise ValueError(f"Empty codelist: {', '.join(empty)}")

    def validate(self, df: IamDataFrame, dimensions: list = None) -> None:
        """Validate that the coordinates of `df` are defined in the codelists

        Parameters
        ----------
        df : :class:`pyam.IamDataFrame`
            Scenario data to be validated against the codelists of this instance.
        dimensions : list of str, optional
            Dimensions to perform validation (defaults to all dimensions of self)

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If `df` fails validation against any codelist.
        """
        validate(self, df, dimensions=dimensions or self.dimensions)

    def check_aggregate(self, df: IamDataFrame, **kwargs) -> None:
        """Check for consistency of scenario data along the variable hierarchy

        Parameters
        ----------
        df : :class:`pyam.IamDataFrame`
            Scenario data to be checked for consistency along the variable hierarchy.
        kwargs : Tolerance arguments for comparison of values
            Passed to :any:`numpy.isclose` via :any:`pyam.IamDataFrame.check_aggregate`.

        Returns
        -------
        :class:`pandas.DataFrame` or None
            Data where a variable and its computed aggregate does not match.

        Raises
        ------
        ValueError
            If the :any:`DataStructureDefinition` does not have a *variable* dimension.
        """
        if "variable" not in self.dimensions:
            raise ValueError("Aggregation check requires 'variable' dimension.")

        lst = []

        with adjust_log_level(level="WARNING"):
            for code in df.variable:
                attr = self.variable.mapping[code]
                if attr.check_aggregate:
                    components = attr.components

                    # check if multiple lists of components are given for a code
                    if isinstance(components, dict):
                        for name, _components in components.items():
                            error = df.check_aggregate(code, _components, **kwargs)
                            if error is not None:
                                error.dropna(inplace=True)
                                # append components-name to variable column
                                error.index = replace_index_labels(
                                    error.index, "variable", [f"{code} [{name}]"]
                                )
                                lst.append(error)

                    # else use components provided as single list or pyam-default (None)
                    else:
                        error = df.check_aggregate(code, components, **kwargs)
                        if error is not None:
                            lst.append(error.dropna())

        if lst:
            # there may be empty dataframes due to `dropna()` above
            error = pd.concat(lst)
            return error if not error.empty else None

    def to_excel(
        self, excel_writer, sheet_name=None, sort_by_code: bool = False, **kwargs
    ):
        """Write the *variable* codelist to an Excel sheet

        Parameters
        ----------
        excel_writer : path-like, file-like, or ExcelWriter object
            File path as string or :class:`pathlib.Path`,
            or existing :class:`pandas.ExcelWriter`.
        sheet_name : str, optional
            Name of sheet that will have the codelist. If *None*, use the codelist name.
        sort_by_code : bool, optional
            Sort the codelist before exporting to file.
        **kwargs
            Passed to :class:`pandas.ExcelWriter` (if *excel_writer* is path-like).
        """
        # TODO write all dimensions to the file
        self.variable.to_excel(excel_writer, sheet_name, sort_by_code, **kwargs)

    def validate_RegionProcessor(self, rp: RegionProcessor) -> None:
        """Check if all mappings are valid and collect all errors."""
        errors = []
        for mapping in rp.mappings.values():
            try:
                self.validate_RegionAggregationMapping(mapping)
            except RegionNotDefinedError as rnde:
                errors.append(ErrorWrapper(rnde, f"mappings -> {mapping.model}"))
        if errors:
            raise pydantic.ValidationError(errors, model=rp.__class__)

    def validate_RegionAggregationMapping(self, ram: RegionAggregationMapping):
        if hasattr(self, "region"):
            if invalid := [
                c for c in ram.all_regions if c not in getattr(self, "region")
            ]:
                raise RegionNotDefinedError(region=invalid, file=ram.file)

    def validate_RequiredDataValidator(self, rdv: RequiredDataValidator) -> None:

        errors = []
        for field, value in (
            (field, getattr(rdv, field))
            for field in ("required_data", "optional_data")
            if getattr(rdv, field) is not None
        ):
            for i, data in enumerate(value):
                try:
                    self.validate_RequiredData(data)
                except ValueError as ve:
                    errors.append(
                        ErrorWrapper(
                            ve,
                            (
                                f"In file {get_relative_path(rdv.file)}\n{field} "
                                f"entry nr. {i+1}"
                            ),
                        )
                    )
        if errors:
            raise pydantic.ValidationError(errors, model=rdv.__class__)

    def validate_RequiredData(self, rd: RequiredData) -> None:
        error_msg = ""

        # check for undefined regions and variables
        for dim in ("region", "variable"):
            if not_defined_dims := self._undefined_dimension(dim, rd):
                error_msg += (
                    f"The following {dim}(s) were not found in the "
                    f"DataStructureDefinition:\n{not_defined_dims}\n"
                )
        # check for defined variables with wrong units
        if wrong_unit_variables := self._wrong_unit_variables(rd):
            error_msg += (
                "The following variables were found in the "
                "DataStructureDefinition but have the wrong unit "
                "(affected variable, wrong unit, expected unit):\n"
                f"{wrong_unit_variables}"
            )

        if error_msg:
            raise ValueError(error_msg)

    def _undefined_dimension(self, dimension: str, rd: RequiredData) -> List[str]:
        missing_items: List[str] = []
        # only check if both the current instance and the DataStructureDefinition
        # have the dimension in question
        if getattr(rd, dimension) and hasattr(self, dimension):
            missing_items.extend(
                dim
                for dim in getattr(rd, dimension)
                if dim not in getattr(self, dimension)
            )

        return missing_items

    def _wrong_unit_variables(self, rd: RequiredData) -> List[Tuple[str, str, str]]:
        wrong_units: List[Tuple[str, str, str]] = []
        if hasattr(self, "variable"):
            wrong_units.extend(
                (var, rd.unit, getattr(self, "variable")[var].unit)
                for var in getattr(self, "variable")
                if rd.unit and rd.unit not in getattr(self, "variable")[var].units
            )

        return wrong_units
