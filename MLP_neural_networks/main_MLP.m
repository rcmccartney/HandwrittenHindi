
% run convert_normalize3 if necessary to make the data

clc; clear;

% GLOBAL PARAMETERS FOR ALL DATASETS
baseDir='C:\Users\mccar_000\Desktop\neural_nets';
opts.lambda = 1.0;
opts.hidden_layer_size = 25;
opts.iterations = 50;
opts.batchsize = 100;
opts.learning_rate = 0.01;
opts.momentum = 0.9;
opts.init_wt = 0.01;

% ========================== MNIST ========================================

diary( 'output\MNIST' );
opts.num_labels = 10;  % 0 is 10
opts.name = 'mnist';
imdb = load(fullfile(baseDir, 'data', 'mnist.mat'));

% this runs advanced optimization on MNIST
% train_fmincg(imdb, opts);

% this runs backprop with momentum
data.trainData = [ imdb.data, imdb.labels];
data.validData = [ imdb.data, imdb.labels];
data.testData = [ imdb.test, imdb.test_labels];
train_backprop_crossEntropy(data, opts);

% ========================== DEVNAGARI ====================================

diary( 'output\DEVNAGARI' );
opts.num_labels = 111;  
opts.name = 'devnagari';
imdb = load(fullfile(baseDir, 'devnagari_normalized', 'devnagari_normalized.mat'));

% train_fmincg(imdb, opts);
data.trainData = [ imdb.data, imdb.labels];
data.validData = [ imdb.data, imdb.labels];
data.testData = [ imdb.test, imdb.test_labels];
train_backprop_crossEntropy(data, opts);

% ========================== TAMIL ========================================

diary( 'output\TAMIL' );
opts.num_labels = 156;
opts.name = 'tamil';
imdb = load(fullfile(baseDir, 'tamil_normalized', 'tamil_normalized.mat'));

% train_fmincg(imdb, opts);
data.trainData = [ imdb.data, imdb.labels];
data.validData = [ imdb.data, imdb.labels];
data.testData = [ imdb.test, imdb.test_labels];
train_backprop_crossEntropy(data, opts);

% ========================== TELUGU =======================================

diary( 'output\TELUGU' );
opts.num_labels = 169;
opts.name = 'telugu';
imdb = load(fullfile(baseDir, 'telugu_normalized', 'telugu_normalized.mat'));

% train_fmincg(imdb, opts);
data.trainData = [ imdb.data, imdb.labels];
data.validData = [ imdb.data, imdb.labels];
data.testData = [ imdb.test, imdb.test_labels];
train_backprop_crossEntropy(data, opts);