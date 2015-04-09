% =========================================================================
% ========================== RBF ==========================================

% Add the subdirectories to the path.
addpath('kMeans');
addpath('RBFN');
baseDir='C:\Users\mccar_000\Desktop\neural_nets';

% ========================== MNIST ========================================

diary( 'output\MNIST_RBF' );
imdb = load(fullfile(baseDir, 'data', 'mnist.mat'));
% have to set how many clusters you want to use as prototypes for 
% each class, this multiplied by the number of classes is the number
% of RBF nodes in the hidden layer
opts.clusters = 1;
opts.num_labels = 10;  % 0 is 10
run_RBFN(imdb, opts);

% ========================== DEVNAGARI ====================================


diary( 'output\DEVNAGARI_RBF' );
imdb = load(fullfile(baseDir, 'devnagari_normalized', 'devnagari_normalized.mat'));
opts.clusters = 3;
opts.num_labels = 111;  
run_RBFN(imdb, opts);

% ========================== TAMIL ========================================

diary( 'output\TAMIL_RBF' );
imdb = load(fullfile(baseDir, 'tamil_normalized', 'tamil_normalized.mat'));
opts.clusters = 1;
opts.num_labels = 156;
run_RBFN(imdb, opts);

% ========================== TELUGU =======================================

diary( 'output\TELUGU_RBF' );
imdb = load(fullfile(baseDir, 'telugu_normalized', 'telugu_normalized.mat'));
opts.clusters = 1;
opts.num_labels = 169;
run_RBFN(imdb, opts);
