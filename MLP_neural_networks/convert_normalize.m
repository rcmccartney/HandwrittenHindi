%Sagar Barbhaya, Jie Yuan, Robert McCartney
%Preprocessing of MNIST and Indian language data sets
%March 2, 2015

% SET YOUR PATH HERE
opts.baseDir='C:\Users\mccar_000\Desktop\neural_nets';
opts.dataDir = fullfile(opts.baseDir, 'data');
opts.mnistPath = fullfile(opts.dataDir, 'mnist.mat');
% percent of data to use for training
testfrac = 0.7; 

% ========================  MNIST PORTION =================================

%if ~exist(opts.mnistPath, 'file')
%  imdb = getMnist(opts) ;
%  save(opts.mnistPath, '-struct', 'imdb') ;
%end

% =========================================================================


% ========================  INDIAN PORTION ================================
originalFiles = {'hpl-devnagari-iso-char-offline' ...
    'hpl-telugu-iso-char-offline' 'hpl-tamil-iso-char-offline'};
normalizedFiles = {'devnagari_normalized' 'telugu_normalized' 'tamil_normalized'};

for dataSetIndex = 1:3,
    fullfile(opts.baseDir, normalizedFiles{dataSetIndex})
	if ~exist(fullfile(opts.baseDir, normalizedFiles{dataSetIndex}), 'dir'),
		disp(['Need to normalize the data for ', char(originalFiles{dataSetIndex})])
		originalFile = char(originalFiles{dataSetIndex});
		normalizedFile = char(normalizedFiles{dataSetIndex});
		
		cd(originalFile)
		files = dir();
		newDir=strcat(opts.baseDir, '\', normalizedFile);
		if ~exist(newDir,'dir')
			mkdir(newDir)
		end
		for file = files'
			name = getfield(file,'name'); %user1, user2, ...
			if ~(strcmp(name,'.') || strcmp(name,'..'))
				disp(['processing dir: ', name])
				cd(name)
				imFiles = dir('*.tiff');
				for imFile = imFiles'
					imName = getfield(imFile,'name');
					if ~(strcmp(imName,'.') || strcmp(imName,'..'))
						 nameParts = strsplit(imName,'t');
						 cellNum = char(nameParts(1));
						 cellDir = strcat(newDir,'\',cellNum);
						 if ~exist(cellDir)
							mkdir(cellDir)
						 end
						 num_dir = length(dir(strcat(cellDir,'\*.tiff')));
						 cellFileName = strcat(cellNum,'j',num2str(num_dir),'.tiff');
						 newFileDir = strcat(cellDir,'\',cellFileName);
						 copyfile(imName, newFileDir)
					end
				end
				cd ..
			end
		end
		cd ..

		cd(normalizedFile)
		files = dir();
		files(~[files.isdir]) = [];
		for file = files'
			name = getfield(file, 'name'); %gesture 000, e.g.
			%nameParts = strsplit(imName,'j');
			%charNum = char(nameParts(1));
			charNum = char(name);
			%charMatrix = genvarname(['devNorm' charNum]); %generate
			%devnagari_normalized_000 e.g.
			%if ~exist(charMatrix,'var')
			%    eval([charMatrix '=zeros(50,28*28);']); %create dataMatrix
			%end
			charMatrix = zeros(50,28*28);
			if ~(strcmp(name,'.') || strcmp(name,'..'))
				cd(name)
				imFiles = dir('*.tiff');
				charMatrixIndex = 0;
				for imFile = imFiles'
					imName = getfield(imFile,'name'); %001j01.tiff, e.g.
					if ~(strcmp(imName,'.') || strcmp(imName,'..'))
						A=imread(imName);
						A2=1-A;
						B=imresize(A2,[28,28]);
						B2=reshape(B',28*28,1); %784x1
						B2=B2';
						B3=(B2-mean(B2))/std(B2);
						charMatrixIndex = charMatrixIndex + 1;
						%eval([charMatrix '(charMatrixIndex,:) = B3;']);
						charMatrix(charMatrixIndex,:) = B3;
					end
				end
				cd .. %puts the files into devnagari_normalized
				disp(['processed: ' charNum])
				charMatrix(all(charMatrix==0,2),:)=[];
				dlmwrite(strcat(normalizedFile,'_',charNum,'.ascii'),charMatrix)
				%eval(['dlmwrite(strcat(''devnagari_normalized_'',charNum,''.ascii''),' charMatrix ')']);
			end
		end
		cd ..
	end;
end;

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
