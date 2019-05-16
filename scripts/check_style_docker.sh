#!/usr/bin/env sh

# Good and locked to a version: origin/develop
CLANG_FORMAT=unibeautify/clang-format

set -evx

docker run --rm ${CLANG_FORMAT} --version
docker run --rm --user=$(id -u):$(id -g) -v "$(pwd)":/workdir -w /workdir ${CLANG_FORMAT} -style=file -sort-includes -i $(git ls-files -- '*.cpp' '*.hpp' '*.cu' '*.h')

git diff --exit-code --color

set +evx
