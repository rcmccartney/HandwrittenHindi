function [net, info] = telugu(varargin)
% Added a third layer of dropout from Run2

run(fullfile('C:\Users\Henry\Box Sync\Projects\matconvnet-master\matlab', ...
    'vl_setupnn.m')) ;

opts.dataDir = fullfile('data');
opts.expDir = fullfile(opts.dataDir, 'telugu');
opts.loadMat = fullfile(opts.expDir, 'telugu.mat');
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.train.batchSize = 200 ;
opts.train.numEpochs = 100 ;  
opts.train.continue = true ;  % can continue training after stopping
opts.train.useGpu = true ;
opts.train.learningRate = [0.1*ones(1, 25) 0.01*ones(1, 25) 0.001*ones(1, 25) 0.0001*ones(1,25)] ;
opts.weightDecay = 0.0005 ;
opts.momentum = 0.5 ;
opts.train.outputClasses = 169 ;
opts.train.expDir = opts.expDir ;
opts = vl_argparse(opts, varargin) ;

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------

if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = getImdb(opts) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

f=1/100 ;
net.layers = {} ;
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(5,5,1,20, 'single'), ...
                           'biases', zeros(1, 20, 'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(5,5,20,50, 'single'),...
                           'biases', zeros(1,50,'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(4,4,50,500, 'single'),...
                           'biases', zeros(1,500,'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(1,1,500,opts.train.outputClasses, 'single'),...
                           'biases', zeros(1,opts.train.outputClasses ,'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'softmaxloss') ;

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------
tic ; 
% Take the mean out and make GPU if needed
if opts.train.useGpu
  imdb.images.data = gpuArray(imdb.images.data) ;
end

[net, info] = cnn_train(net, imdb, @getBatch, ...
    opts.train, ...
    'val', find(imdb.images.set == 3)) ;

toc
end

% --------------------------------------------------------------------
function [im, labels] = getBatch(imdb, batch)
% --------------------------------------------------------------------
im = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;
end 
% --------------------------------------------------------------------
function imdb = getImdb(opts)
% --------------------------------------------------------------------

% get the matrix with data
processedData = load(fullfile(opts.loadMat));
% get size of train and test
trainsize = size(processedData.data,1) ;
testsize = size(processedData.test,1) ;

% the data is in rows, convert to 2D matrices 
% and add a space for convolutions
for i=1:size(processedData.data,1),
    data_row = reshape(processedData.data(i,:),28,28 ) ; 
    data(:,:,1,i) = data_row ;
end;
for i=1:size(processedData.test,1),
    test_row = reshape(processedData.test(i,:),28,28 ) ; 
    test(:,:,1,i) = test_row ;
end;

data = single(cat(4, data, test)) ;
output = [ processedData.labels' processedData.test_labels' ];

% set is a row of ones then threes used by library for training and test sets
% a two would be validation set, not used here
set = [ones(1,trainsize) 3*ones(1,testsize)];

% get the mean of each image so it can be subtracted
% and divide by the std dev
dataMean = mean(data(:,:,:,set == 1), 4);
dataStd = std(data(:,:,:,set == 1), 0, 4);
data = bsxfun(@minus, data, dataMean) ;
data = bsxfun(@rdivide, data, dataStd) ;
imdb.images.data = data;
imdb.images.data_mean = dataMean;
imdb.images.labels = output;
imdb.images.set = set;
imdb.meta.sets = {'train', 'val', 'test'};
imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),1:5,'uniformoutput',false); 
end 