from setuptools import setup, find_packages


setup(name='TVLAD',
      version='1.0',
      description='Open-source toolbox for Image-based Localization (Place Recognition)',
      author_email='cvprw_22@163.com',
      url='https://github.com/cvprw-22/TVLAD',
      license='MIT',
      install_requires=[
          'numpy', 'torch', 'torchvision',
          'six', 'h5py', 'Pillow', 'scipy',
          'scikit-learn', 'metric-learn'],
      packages=find_packages(),
      keywords=[
          'Image Localization',
          'Image Retrieval',
          'Place Recognition'
      ])
