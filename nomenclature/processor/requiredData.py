import logging
from pathlib import Path
from typing import List, Optional, Union, Tuple

import yaml
from pyam import IamDataFrame
import pydantic
from pydantic import BaseModel, validator
from pydantic.error_wrappers import ErrorWrapper

from nomenclature.definition import DataStructureDefinition
from nomenclature.processor import Processor
from nomenclature.processor.utils import get_relative_path
from nomenclature.error.requiredData import RequiredDataMissingError

logger = logging.getLogger(__name__)


class RequiredData(BaseModel):

    variable: List[str]
    region: Optional[List[str]]
    year: Optional[List[int]]
    unit: Optional[str]

    @validator("variable", "region", "year", pre=True)
    def single_input_to_list(cls, v):
        return v if isinstance(v, list) else [v]

    def validate_with_definition(self, dsd: DataStructureDefinition) -> None:
        error_msg = ""

        # check for undefined regions and variables
        for dim in ("region", "variable"):
            if not_defined_dims := self._undefined_dimension(dim, dsd):
                error_msg += (
                    f"The following {dim}(s) were not found in the "
                    f"DataStructureDefinition:\n{not_defined_dims}\n"
                )
        # check for defined variables with wrong units
        if wrong_unit_variables := self._wrong_unit_variables(dsd):
            error_msg += (
                "The following variables were found in the "
                "DataStructureDefinition but have the wrong unit "
                "(affected variable, wrong unit, expected unit):\n"
                f"{wrong_unit_variables}"
            )

        if error_msg:
            raise ValueError(error_msg)

    def _undefined_dimension(
        self, dimension: str, dsd: DataStructureDefinition
    ) -> List[str]:
        missing_items: List[str] = []
        # only check if both the current instance and the DataStructureDefinition
        # have the dimension in question
        if getattr(self, dimension) and hasattr(dsd, dimension):
            missing_items.extend(
                dim
                for dim in getattr(self, dimension)
                if dim not in getattr(dsd, dimension)
            )

        return missing_items

    def _wrong_unit_variables(
        self, dsd: DataStructureDefinition
    ) -> List[Tuple[str, str, str]]:
        wrong_units: List[Tuple[str, str, str]] = []
        if hasattr(dsd, "variable"):
            wrong_units.extend(
                (var, self.unit, getattr(dsd, "variable")[var].unit)
                for var in getattr(self, "variable")
                if self.unit and self.unit not in getattr(dsd, "variable")[var].units
            )

        return wrong_units


class RequiredDataValidator(Processor):

    name: str
    required_data: List[RequiredData]
    optional_data: Optional[List[RequiredData]]
    file: Path

    @classmethod
    def from_file(cls, file: Union[Path, str]) -> "RequiredDataValidator":
        with open(file, "r") as f:
            content = yaml.safe_load(f)
        return cls(file=file, **content)

    def apply(self, df: IamDataFrame) -> IamDataFrame:
        error = False
        # check for required data and raise error if missing
        for data in self.required_data:
            if (missing_index := df.require_data(**data.dict())) is not None:
                error = True
                logger.error(
                    f"Required data {data} from file {get_relative_path(self.file)} "
                    f"missing for:\n{missing_index}"
                )
        if error:
            raise RequiredDataMissingError(
                "Required data missing. Please check the log for details."
            )

        # check for optional data and issue warning if missing
        if self.optional_data:
            for data in self.optional_data:
                if (missing_index := df.require_data(**data.dict())) is not None:
                    logger.warning(
                        f"Optional data {data} from file "
                        f"{get_relative_path(self.file)} missing for:\n{missing_index}"
                    )
        return df

    def validate_with_definition(self, dsd: DataStructureDefinition) -> None:

        errors = []
        for field, value in (
            (field, getattr(self, field))
            for field in ("required_data", "optional_data")
            if getattr(self, field) is not None
        ):
            for i, data in enumerate(value):
                try:
                    data.validate_with_definition(dsd)
                except ValueError as ve:
                    errors.append(
                        ErrorWrapper(
                            ve,
                            (
                                f"In file {get_relative_path(self.file)}\n{field} "
                                f"entry nr. {i+1}"
                            ),
                        )
                    )
        if errors:
            raise pydantic.ValidationError(errors, model=self.__class__)