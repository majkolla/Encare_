# ============================================================
# file: 01_explore_raw.R
# project: encare
#
# inputs:
# data/raw/your_data.csv
#
# notes:
# this script should not modify or overwrite raw data
# ============================================================

# packages
library(tidyverse)
library(janitor)
library(skimr)
library(readr)
library(purrr)

# paths
data_path <- "data/data.csv"

# read data
# makes column names consistent and easy to use in R (snake_case, lowercase);
# this ONLY renames columns, all data values stay unchanged.
raw <- readr::read_csv(data_path, show_col_types = FALSE) |>
  janitor::clean_names()

problems(raw)
# basic structure
glimpse(raw)
dim(raw)

profile <- tibble(
  var = names(raw),
  class = map_chr(raw, ~ class(.x)[1]),
  n_missing = map_int(raw, ~ sum(is.na(.x))),
  pct_missing = map_dbl(raw, ~ mean(is.na(.x))),
  n_unique = map_int(raw, ~ dplyr::n_distinct(.x, na.rm = TRUE)),
  sample_values = map_chr(raw, ~ paste(head(unique(na.omit(.x)), 8), collapse = " | "))
)

print(profile, n = Inf)

probs_named <- problems(raw) |>
  mutate(var = names(raw)[col])

print(probs_named, n = Inf)


low_cardinality <- tibble(
  var = names(raw),
  class = map_chr(raw, ~ class(.x)[1]),
  n_unique = map_int(raw, ~ dplyr::n_distinct(.x, na.rm = TRUE)),
  values = map(raw, ~ sort(unique(na.omit(.x))))
) |>
  filter(n_unique <= 8)

print(low_cardinality, n = Inf, width = Inf)


# column names
names(raw)

# quick summary
summary(raw)

# skim overview (types, missingness, distributions)
skim(raw)

# check for duplicate rows
raw |>
  summarise(n_rows = n(),
            n_distinct_rows = n_distinct(raw))

# numeric variable exploration
raw |>
  select(where(is.numeric)) |>
  pivot_longer(everything()) |>
  ggplot(aes(value)) +
  geom_histogram(bins = 30) +
  facet_wrap(~ name, scales = "free") +
  theme_minimal()

# categorical variable exploration
raw |>
  select(where(is.character)) |>
  pivot_longer(everything()) |>
  count(name, value, sort = TRUE)
