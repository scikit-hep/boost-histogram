#!/usr/bin/env sh

set -evx

clang-format --version
git ls-files -- '*.cpp' '*.hpp' '*.cu' '*.h' | xargs clang-format -style=file -sort-includes -i

git diff --exit-code --color

set +evx
