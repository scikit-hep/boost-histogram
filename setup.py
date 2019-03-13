from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools

__version__ = '0.0.1'

ext_modules = [
    Extension(
        'histogram',
        ['src/module.cpp',
         'src/histogram/axis.cpp',
         'src/histogram/histogram.cpp',
         'src/histogram/storage.cpp'
        ],
        include_dirs=[
            'include',
            'extern/assert/include',
            'extern/callable_traits/include',
            'extern/config/include',
            'extern/container_hash/include',
            'extern/core/include',
            'extern/detail/include',
            'extern/histogram/include',
            'extern/integer/include',
            'extern/iterator/include',
            'extern/move/include',
            'extern/mp11/include',
            'extern/mpl/include',
            'extern/preprocessor/include',
            'extern/pybind11/include',
            'extern/static_assert/include',
            'extern/throw_exception/include',
            'extern/type_index/include',
            'extern/type_traits/include',
            'extern/utility/include',
            'extern/variant/include',
        ],
        language='c++'
    ),
]

# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++14 compiler flag.
    """
    if has_flag(compiler, '-std=c++14'):
        return '-std=c++14'
    else:
        raise RuntimeError('Unsupported compiler -- at least C++14 support '
                           'is needed!')


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }

    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)

setup(
    name='histogram',
    version=__version__,
    author='Henry Schreiner',
    author_email='hschrein@cern.ch',
    url='https://github.com/scikit-hep/boost-histogram',
    description='The Boost::Histogram Python wrapper.',
    long_description='',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
    tests_require=['pytest', 'numpy']
)
