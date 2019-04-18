#!/usr/bin/env sh

set -evx

# Also good but untagged: CLANG_FORMAT=unibeautify/clang-format
CLANG_FORMAT=saschpe/clang-format:5.0.1


if [ -x "$(command -v clang-format)" ] ; then

    clang-format --version
    git ls-files -- '*.cpp' '*.hpp' '*.cu' '*.h' | xargs clang-format -style=file -sort-includes -i

elif [ -x "$(command -v docker)" ] ; then

    docker run -it --rm ${CLANG_FORMAT} --version
    docker run -it --rm -v "$(pwd)":/workdir -w /workdir  ${CLANG_FORMAT} -style=file -sort-includes -i $(git ls-files -- '*.cpp' '*.hpp' '*.cu' '*.h')

fi

git diff --exit-code --color

set +evx

