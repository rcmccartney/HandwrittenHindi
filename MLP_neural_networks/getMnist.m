% --------------------------------------------------------------------
function imdb = getMnist(opts)
% --------------------------------------------------------------------
% Prepare the mnist data, returns image data with mean image subtracted
files = {'train-images-idx3-ubyte', ...
         'train-labels-idx1-ubyte', ...
         't10k-images-idx3-ubyte', ...
         't10k-labels-idx1-ubyte'} ;

if ~exist(opts.dataDir, 'dir')
  mkdir(opts.dataDir) ;
end

for i=1:4
  if ~exist(fullfile(opts.dataDir, files{i}), 'file')
    url = sprintf('http://yann.lecun.com/exdb/mnist/%s.gz',files{i}) ;
    fprintf('downloading %s\n', url) ;
    gunzip(url, opts.dataDir) ;
  end
end

f=fopen(fullfile(opts.dataDir, 'train-images-idx3-ubyte'),'r') ;
x1=fread(f,inf,'uint8');
fclose(f) ;
x1=permute(reshape(x1(17:end),28,28,60e3),[2 1 3]) ;
x1 = reshape(x1, 784, [])';

f=fopen(fullfile(opts.dataDir, 't10k-images-idx3-ubyte'),'r') ;
x2=fread(f,inf,'uint8');
fclose(f) ;
x2=permute(reshape(x2(17:end),28,28,10e3),[2 1 3]) ;
x2 = reshape(x2, 784, [])';

f=fopen(fullfile(opts.dataDir, 'train-labels-idx1-ubyte'),'r') ;
y1=fread(f,inf,'uint8');
fclose(f) ;
y1=double(y1(9:end)')+1 ;

f=fopen(fullfile(opts.dataDir, 't10k-labels-idx1-ubyte'),'r') ;
y2=fread(f,inf,'uint8');
fclose(f) ;
y2=double(y2(9:end)')+1 ;

fprintf('Normalizing the MNIST data\n') ;
% subtracting the mean of a single image, not 
% mean of every image pixel over the dataset
% dataMean = repmat(mean(x1, 2), 1, 784) ;
% dataStd = std(x1);
% data = x1 - dataMean ;
% data = bsxfun(@rdivide, data, dataStd);
% do same to test set
% dataMean = repmat(mean(x2, 2), 1, 784) ;
% test = x2 - dataMean ;
% test = bsxfun(@rdivide, test, dataStd);

dataMean = mean(x1);
data = bsxfun(@minus, x1, dataMean);
test = bsxfun(@minus, x2, dataMean);

imdb.data = data ;
imdb.test = test ;
imdb.data_mean = dataMean ;
imdb.labels = y1' ; 
imdb.test_labels = y2' ;