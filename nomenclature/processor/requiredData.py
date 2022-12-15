import logging
from pathlib import Path
from typing import List, Optional, Union

import yaml
from pyam import IamDataFrame
from pydantic import BaseModel, validator

from nomenclature.error.requiredData import RequiredDataMissingError
from nomenclature.processor.utils import get_relative_path

logger = logging.getLogger(__name__)


class RequiredData(BaseModel):

    variable: List[str]
    region: Optional[List[str]]
    year: Optional[List[int]]
    unit: Optional[str]

    @validator("variable", "region", "year", pre=True)
    def single_input_to_list(cls, v):
        return v if isinstance(v, list) else [v]


class RequiredDataValidator(BaseModel):

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
