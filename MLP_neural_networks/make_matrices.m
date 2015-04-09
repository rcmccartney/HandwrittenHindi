normalizedFiles = {'teluguNormalized' 'tamilNormalized' 'devnagariNormalized' };

opts.baseDir= 'C:\Users\Henry\Box Sync\NVIDIA\math\Neural Networks';
opts.dataDir = fullfile(opts.baseDir, 'data');

testfrac = 0.7 ;

% Now create the actual matrices
for i=1:length(normalizedFiles),
    index = 1;
    trainData = [];
    output = [];
    if ~exist(fullfile(opts.baseDir, normalizedFiles{i}, ...
            strcat(normalizedFiles{i},'.mat')), 'file'),
        sprintf('#### Starting %s ####',normalizedFiles{i})
        files = dir(fullfile(opts.baseDir, normalizedFiles{i}));
        fileNames = {files(~[files.isdir]).name};
        for k=1:length(fileNames),
            disp(['processing dir: ', fileNames{k}])
            images = load(fullfile(opts.baseDir, normalizedFiles{i}, fileNames{k}));
            trainData = [trainData; images];
            labels = ones(size(images, 1), 1) * index;
            output = [output; labels];
            index = index + 1;
        end;
        % mix up the classes 
        shuffle = randperm(size(output,1));
        output = output(shuffle);
        trainData = trainData(shuffle, :);
        % split into train and test
        trainsize = int64(testfrac*size(output,1));
        imdb.data = trainData(1:trainsize, :);
        imdb.labels = output(1:trainsize, :);
        imdb.test = trainData(trainsize+1:end, :);
        imdb.test_labels = output(trainsize+1:end, :);
        save(fullfile(opts.baseDir, normalizedFiles{i}, normalizedFiles{i}), '-struct', 'imdb') ;
    end;
end;