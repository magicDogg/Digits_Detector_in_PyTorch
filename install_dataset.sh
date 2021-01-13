mkdir data
cd data

wget http://ufldl.stanford.edu/housenumbers/train_32x32.mat
wget http://ufldl.stanford.edu/housenumbers/test_32x32.mat
wget https://www.cs.toronto.edu/~kriz/cifar-100-matlab.tar.gz

tar -xf cifar-100-matlab.tar.gz && rm -f cifar-100-matlab.tar.gz
cd cifar-100-matlab
mv train.mat ../
mv test.mat ../
rm meta.mat
cd ../
rmdir cifar-100-matlab

