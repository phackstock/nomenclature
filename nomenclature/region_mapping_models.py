from pathlib import Path
from typing import Dict, List, Optional, Union
from collections import Counter
from pydantic.types import FilePath

import yaml
from jsonschema import validate, ValidationError
import pydantic
from pydantic import BaseModel, validator, root_validator, PydanticValueError

import copy

# We take a deep copy of the original __str__ from
# pydantic.error_wrappers.ValidationError. We do this to keep the changes minimally
# invasive and get 'automatic' updates in case of any changes upstream
original__str__ = copy.deepcopy(pydantic.error_wrappers.ValidationError.__str__)

# Define a new __str__ method which adds file information in case it is present.
# Otherwise the original __str__ method is used.
def new__str__(self):
    """Change __str__ from pydantic ValidationError to include the file name if
    present"""
    if "ctx" in self.errors()[0] and "file" in self.errors()[0]["ctx"]:
        return original__str__(self).replace(
            "\n", f" in {self.errors()[0]['ctx']['file']}\n", 1
        )
    return original__str__(self)


# Overwrite the original __str__ with new__str__
pydantic.error_wrappers.ValidationError.__str__ = new__str__

# Custom error class since we need to get the file information to ValidationError
# See for details: https://pydantic-docs.helpmanual.io/usage/models/#custom-errors
class DoubleNativeRegionError(PydanticValueError):
    code = "double_native_region"
    msg_template = 'Two or more native regions share the same name: "{duplicates}"'


here = Path(__file__).parent.absolute()


class NativeRegion(BaseModel):
    name: str
    rename: Optional[str]

    @property
    def target_native_region(self):
        return self.rename if self.rename is not None else self.name


class CommonRegion(BaseModel):
    name: str
    constituent_regions: List[NativeRegion]


class RegionAggregationMapping(BaseModel):
    model: str
    file: FilePath
    native_regions: Optional[List[NativeRegion]]
    common_regions: Optional[List[CommonRegion]]

    @validator("native_regions")
    def validate_native_regions(cls, v, values):
        target_names = [nr.target_native_region for nr in v]
        duplicates = [
            item for item, count in Counter(target_names).items() if count > 1
        ]
        if duplicates:
            # Raise the custom DoubleNativeRegionError and give the parameters
            # duplicates and file.
            raise DoubleNativeRegionError(
                duplicates=duplicates, file=values["file"].relative_to(Path.cwd())
            )
        return v

    @validator("common_regions")
    def validate_common_regions(cls, v, values):
        names = [cr.name for cr in v]
        duplicates = [item for item, count in Counter(names).items() if count > 1]
        if duplicates:
            raise ValueError(
                f"Duplicated aggregation mapping to common regions: {duplicates} in "
                f"{values['file'].relative_to(Path.cwd())}"
            )
        return v

    @root_validator()
    def check_illegal_renaming(cls, values):
        """Check if any renaming overlaps with common regions"""
        # Skip if only either native-regions or common-regions are specified
        if values.get("native_regions") is None or values.get("common_regions") is None:
            return values
        native_region_names = {
            nr.target_native_region for nr in values["native_regions"]
        }
        common_region_names = {cr.name for cr in values["common_regions"]}
        overlap = list(native_region_names & common_region_names)
        if overlap:
            raise ValueError(
                "Conflict between (renamed) native regions and aggregation mapping"
                f" to common regions: {overlap} in "
                f"{values['file'].relative_to(Path.cwd())}"
            )
        return values

    @classmethod
    def create_from_region_mapping(cls, file: Union[Path, str]):
        SCHEMA_FILE = here / "validation_schemas" / "region_mapping_schema.yaml"
        with open(file, "r") as f:
            mapping_input = yaml.safe_load(f)
        with open(SCHEMA_FILE, "r") as f:
            schema = yaml.safe_load(f)

        # Validate the input data using jsonschema
        try:
            validate(mapping_input, schema)
        except ValidationError as e:
            # Add file information in case of error
            raise ValidationError(f"{e.message} in {file}")

        # Add the file name to mapping_input
        mapping_input["file"] = file

        # Reformat the "native_regions"
        if "native_regions" in mapping_input:
            native_region_list: List[Dict] = []
            for nr in mapping_input["native_regions"]:
                if isinstance(nr, str):
                    native_region_list.append({"name": nr})
                elif isinstance(nr, dict):
                    native_region_list.append(
                        {"name": list(nr)[0], "rename": list(nr.values())[0]}
                    )
            mapping_input["native_regions"] = native_region_list

        # Reformat the "common_regions"
        if "common_regions" in mapping_input:
            common_region_list: List[Dict[str, List[Dict[str, str]]]] = []
            for cr in mapping_input["common_regions"]:
                cr_name = list(cr)[0]
                common_region_list.append(
                    {
                        "name": cr_name,
                        "constituent_regions": [{"name": x} for x in cr[cr_name]],
                    }
                )
            mapping_input["common_regions"] = common_region_list
        return cls(**mapping_input)


# Just as a demonstration for the error message, to be removed later
if __name__ == "__main__":
    RegionAggregationMapping.create_from_region_mapping(
        Path(__file__).parents[1]
        / "tests/data/region_aggregation/illegal_mapping_duplicate_native.yaml"
    )