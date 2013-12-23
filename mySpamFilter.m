classdef mySpamFilter < myNaiveBayes
    %mySpamFilter extends myNaiveBayes class with file processing
    %   This is  a subclass of myNaiveBayes and adds an ability to
    %   use dataset from SpamAssasin.
    % 
    %   The source files are grouped in three folders: 'easy_ham', 
    %   'hard_ham', and 'spam', and they are enclosed in a directory
    %   you specify 'data_src' as string.
    %
    %   Instatiate mySpamFilter object
    %
    %       nb = mySpamFilter();
    %   
    %   Build the dataset from files from a directory
    %   based on a specified training set split ratio which
    %   random mix of samples, unless specified to repeat the 
    %   same split. 
    %   
    %   Default: 'data_src' = 'ds_reduced', split = 0.7,
    %            repeat = false
    % 
    %       nb.buildDataset('data_src', split, repeat);
    %
    %   Build the prediction model from the training set and
    %   evaluate it with test set
    %
    %       nb.buildModel();
    %
    %   Finally, classify a single email from a text file
    %
    %       predicted_class = nb.classifyEmail('emailSample1.txt')
    
    properties
        vocab;      % list of words in vocabulary
        Xtrain;     % training set - predictors
        ytrain;     % training set - response
        Xtest;      % test set     - predictors
        ytest;      % test set     - response
    end
    
    methods
        % buildDataset(self,'data_src',split,repeat)
        %   build training and test sets from files
        % buildModel(self)
        %   build Naive Bayes model from the trainig set;
        %   produce evalutation with the test set
        % evaluate(self)
        %   called by buildModel() to run evaluation
        % predicted_class = classifyEmail(self,filename)
        %   read an email as a text file and classify it
        % [file_contents,labels] = getFileContents(self,ds)
        %   called by buildDataset() to parse the file content
        % fileList = getFileList(~,ds)
        %   called by getFileContents() to generate a list of files
        % file_contents = readFile(~,filename)
        %   called by getFileContents() to read a text file
        % tokens = tokenizeEmail(~,email)
        %   called by getFileContents() to tokenize the text
        % tdf = computeTDF(~,file_contents,word_list)
        %   called by buildDataset() to compute TDF
        % word_list = createWordList(~,file_contents)
        %   called by buildDataset() to create a word list
        
        function buildDataset(self,data_src,split,repeat)
            % define default input args
            switch nargin
                case 3
                    % default - randomize the split
                    repeat = false;
                case 2
                    % default split - 70% training
                    split = 0.7;
                    repeat = false;
                case 1
                    % default data source
                    data_src = 'ds_reduced';
                    split = 0.7;
                    repeat = false;
            end
            
            fprintf('Start building the dataset...\n\n')
            
            % enable parallel computing
            poolobj = gcp('nocreate');
            if isempty(poolobj)
                parpool('local',2);
            end
            
            % get words in emails and class labels
            [file_contents,labels] = getFileContents(self,data_src);
            
            % shuffle the samples to mix spams and hams
            m = length(file_contents);
            if repeat
                % set random number generator to a fixed value
                rng(1)
            end
            randidx = randperm(m);
            file_contents = file_contents(randidx);
            labels = labels(randidx);
            
            % split the training set and test set
            istrain = false(m,1);
            if repeat
                % set random number generator to a fixed value
                rng(1)
            end
            istrain(randperm(m,ceil(m*split))) = true;
            train_contents = file_contents(istrain);
            test_contents = file_contents(~istrain);
            self.ytrain = labels(istrain);
            self.ytest = labels(~istrain);
            fprintf('\nTrainig set: %d of %d files (%.2f%%)...\n\n',...
                sum(istrain),m,split*100)
            tabulate(self.ytrain)
            fprintf('\n')
            
            % create a list of all words
            word_list = createWordList(self,train_contents);
            
            % compute Term Document Frequency
            tdf_train = computeTDF(self,train_contents,word_list);
            tdf_test = computeTDF(self,test_contents,word_list);
            % compute Term Frequency
            tf = sum(tdf_train);
            ignore = false(size(tf));
            % ignore terms that appear only once
            ignore = ignore | tf == 1;
            % ignore terms that appear in all emails
            ignore = ignore | tf == sum(istrain);
            % store the dataset
            self.vocab = word_list(~ignore);
            self.Xtrain = tdf_train(:,~ignore);
            self.Xtest = tdf_test(:,~ignore);
            
            % reset random number generator
            rng default
            
            % shut down parallel computing
            delete(poolobj);
        end
        
        function buildModel(self)
            disp('Using the training set for modeling...')
            train(self,self.Xtrain,self.ytrain);
            evaluate(self);
        end
        
        function evaluate(self)
            disp('Using the test set for evaluation...')
            p = predict(self,self.Xtest);
            y = self.ytest;
            
            fprintf('Checking the prediction performance...\n\n')
            confmat = confusionmat(y,p,'order',[1,0]);
            confmat = confmat./sum(sum(confmat));
            precision = confmat(1,1)/sum(confmat(:,1));
            recall = confmat(1,1)/sum(confmat(1,:));
            f1 = 2*(precision*recall/(precision+recall));
            
            % print the results to the screen
            disp('Test Set Metrics...')
            fprintf('Accuracy: %.2f%%\n',...
                mean(double(p == y)) * 100);
            fprintf('Precision: %.2f%%\n', precision * 100);
            fprintf('Recall: %.2f%%\n', recall * 100);
            fprintf('F1 Score: %.2f%%\n', f1 * 100);
            class = {'spam','ham'};
            fprintf('\nConfusion Matrix\n')
            disp('Actual classes as rows, predicted classes as columns')
            disp(table(confmat(:,1),confmat(:,2),...
                'VariableNames',class,'RowNames',class))
        end
        
        function predicted_class = classifyEmail(self, filename)
            email = readFile(self,filename);
            tokens = tokenizeEmail(self,email);
            [words,~,idx] = unique(tokens);
            counts = accumarray(idx,1);
            file_contents{1}{1} = words;
            file_contents{1}{2} = counts;
            X = computeTDF(self,file_contents,self.vocab);
            predicted_class = predict(self,X);
        end
        
        % Extracts the words in emails and class labels from raw files
        function [file_contents,labels] = getFileContents(self,ds)         
            fileList = getFileList(self,ds);
            m = length(fileList); % total number of samples
            labels = zeros(m,1);
            
            disp('Processing files...')
            tic
            % read file contents
            parfor i = 1:m
                % read content from the file
                path = fullfile(ds,fileList(i).folder,fileList(i).name);
                email = readFile(self,path);
                tokens = tokenizeEmail(self, email)

                % remove duplicates
                [words,~,idx] = unique(tokens);
                counts = accumarray(idx,1);
                file_contents{i} = {words,counts};
                if strcmp(fileList(i).folder,'spam')
                     labels(i) = 1;
                end
                fprintf('.')
                if mod(i,100) == 0
                    fprintf('\nFiles still being processed...')
                end
            end
            
            disp('Done.')
            elapsedTime = toc;
            fprintf('Elapsed time is %f minutes.\n',elapsedTime/60)
            
        end
        
        % Get the file list from the designated source
        function fileList = getFileList(~,ds)
            % email files are stored in three folders
            dirs = {'easy_ham','hard_ham','spam'};
            % initialize a struct to hold list of files
            fileList = struct('name',{},'date',{},'bytes',{},...
                'isdir',{},'datenum',{},'folder',{});
            fprintf('Reading files...')
            % loop over reach folder to get file lists and file counts
            for i = 1:length(dirs)
                % dir returns the list of folder content as a struct array
                s = dir(fullfile(ds,dirs{i}));
                % remove non-file content
                isDir = cat(1,s.isdir); % get folders
                s(isDir) = []; % remove folders
                % remove Mac OSX hidden files
                s = s(arrayfun(@(x) x.name(1),s) ~= '.');
                % remove an extra file that starts with '0000'
                s = s(~strcmp(arrayfun(@(x) x.name(1:4),...
                    s, 'UniformOutput',false),'0000'));
                % add folder field
                [s(:).folder] = deal(dirs{i});
                fileList = [fileList; s];
            end
            m = length(fileList); % total number of samples
            fprintf('%5d files in total\n',m)
        end
        
        % Get the content from a specified file
        function file_contents = readFile(~,filename)
            fid = fopen(filename);
            if fid
                file_contents = fscanf(fid, '%c', inf);
                fclose(fid);
            else
                file_contents = '';
                fprintf('Unable to open %s\n', filename);
            end
        end
        
        % Process an email into stemmed tokens
        function tokens = tokenizeEmail(~,email)
            % lower case
            email = lower(email);
            % strip all HTML
            email = regexprep(email, '<[^<>]+>',' ');
            % handle numbers
            email = regexprep(email, '[0-9]+','number');
            % handle URLs
            email = regexprep(email,...
                '(http|https)://[^\s]*','httpaddr');
            % handle email addresses
            email = regexprep(email,...
                '[^\s]+@[^\s]+','emailaddr');
            % handle $ sign
            email = regexprep(email, '[$]+', 'dollar');
                
            % tokenize content into words and get rid of punctuations
            delims = {' ','@','$','/','#','.','-',':','&','*',...
                '+','=','[',']','?','!','(',')','{','}',',','''',...
                '"','>','_','<',';','%',char(10),char(13)};
            % tokenize the content - returns a nested cell array
            tokens = textscan(email,'%s','Delimiter',delims);
            % flatten the nested cell array
            tokens = tokens{1}(:);
            % remove any non alphanumeric characters
            tokens = regexprep(tokens,'[^a-zA-Z0-9]','');
            % remove empty elements
            tokens = tokens(~cellfun('isempty',tokens));
            % stem the word
            for k = 1:length(tokens)
                try 
                    tokens{k} = porterStemmer(strtrim(tokens{k})); 
                catch
                    tokens{k} = ''; 
                    continue;
                end
            end
            % skip the word if it is too short
            tokens = tokens(~cellfun(@(x) length(x(:))<1,...
                tokens,'UniformOutput',true));
        end

        function tdf = computeTDF(~,file_contents,word_list)
            % initialize the matrix
            m = length(file_contents);
            tdf = zeros(m,length(word_list));
            
            % use parallization when processing large dataset
            if m >= 30
                fprintf('\nComputing word frequencies...\n\n')
                tic
                parfor i = 1:m
                    % temp variable for keeping parfor happy
                    slicerVar = zeros(1,length(word_list));
                    % get the list of words in an email
                    words = file_contents{i}{1};
                    % get the counts of words in an email 
                    counts = file_contents{i}{2};
                    % get the matching indices within the word list
                    [~,cols]=ismember(words,word_list);
                    % if there is any non-match, ignore it
                    counts(cols==0) = [];
                    cols(cols==0) = [];
                    % add the word counts to the matching columns
                    slicerVar(cols) = counts;
                    % copy the result into tdf
                    tdf(i,:) = slicerVar;
                    fprintf('.')
                    if mod(i,60) == 0
                        fprintf('\nStill computing...')
                    end
                end
                elapsedTime = toc;
                disp('Done.')
                fprintf('Elapsed time is %f minutes.\n',elapsedTime/60)
            else % otherwise use regular for loo[
                for i = 1:m
                    words = file_contents{i}{1}; 
                    counts = file_contents{i}{2};
                    [~,cols]=ismember(words,word_list);
                    counts(cols==0) = [];
                    cols(cols==0) = [];
                    tdf(i,cols) = counts;
                end
            end
        end
        
        % Create a list of all words
        function word_list = createWordList(~,file_contents)
            % add up all the words from each email
            word_list = {};
            m = length(file_contents);
            disp('Starting word list creation...')
            tic
            parfor i = 1:m
                words = file_contents{i}{1};
                word_list = [word_list; words];
            end
            % remove duplicates
            word_list = unique(word_list);
            disp('Done.')
            toc
        end
    end
    
end

