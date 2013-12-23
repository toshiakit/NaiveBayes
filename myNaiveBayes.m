classdef myNaiveBayes < handle % subclass 'handle' superclass
    %myNaiveBayes classify spams using Multinomial Naive Bayes Algorithm
    %   Instatiate myNaiveBayes object as follows:
    %       nb = myNaiveBayes();
    %
    %   Train the object
    %       nb.train(trainig_data, class_labels);
    %
    %   Predict with the trained object
    %       predicted_labels = nb.predict(test_data);
    % 
    % * training_data is a numeric feature matrix, 
    % * class_labels is a numeric vector with 1 = spam, 0 = ham
    % * test_data is a feature matrix of the same number of
    %   coumns with training_data.
    % * predicted_labels is a numeric vector with with 1 = spam, 
    %   0 = ham
    
    properties
        prior;      % class prior
        condprob;   % conditonal probabilities of words
    end
    
    methods
        % Constructor
        function self = myNaiveBayes()
        end
        
        % Train the model
        function train(self,training_data,class_labels)
            % This implements multinomial Naive Bayes model.
            % 
            % We want to compute P(class|word) using Bayesian Theorem
            %
            %   p(class|word)= (p(word|class) * 
            %               p(class)) / p(word)
            %
            % but we can ignore the denominator p(word) since we are
            % only interested in relative scores. So we compute:
            %
            %   p'(class|word)= p(word|class) *  p(class)
            %
            % note: 
            % *  p(class)       -> prior probabililty 
            % *  p(word|class)  -> conditional probability 
            % 
            % Using independence assumption, we just multiply the 
            % p(word|class) over al the words in the email to come up
            % with a joint probability p(class|email), but this can 
            % lead to a floating point underflow problem. Solve it 
            % by using log, then multiplication -> summation.
            % 
            %   i.e. log(x*y) = log(x) + log(y) 
            %
            % So the equation changes to:
            %
            %   log(p'(class|word)) = log(p(word|class)) +
            %               log(p(class))
            
            % get the training data and labels
            X = training_data;
            y = class_labels;
            
            % computing prior probabilities...
            %
            %   p(class) = number of emails by class / 
            %               total number of training samples
            
            % get the indices of spam samples
            isspam = y == 1;
            
            % store p(spam) in row1 and p(ham) in row 2
            % don't convert to log yet
            self.prior = [sum(isspam); sum(~isspam)]/length(y);
            
            % computing conditional probabilities for each word...
            % 
            %   p(word|class) = count of word by class / 
            %               total number of words by class
            %
            % but do it in log space, where division -> subtraction
            %   
            %   log(p(word|class)) = log(count of word by class) -  
            %               log(total number of words by class)
            %
            % there is a one problem: log(0) results in error
            % so apply laplace smoothing by adding 1
            % 
            %   log(p(word|class)) = log(count of word by class + 1)  
            %               - log(total number of words by class + 1)
            % 
            % Raplace smoothing in effect adds a baselne probability
            % for words that appear very rarely, so that we don't get
            % completely get zero probability. Instead of 1/1, we can
            % use 1/size of vocabulary to represent that a word occurs
            % at least once for each class.
            
            % store wc of spam in row 1, wc of ham in row 2...
            self.condprob = [sum(X(isspam,:)); sum(X(~isspam,:))];
            % get the total number of words by class
            totalcount = sum(self.condprob,2);
            % convert it into log space, add 1 for laplace smoothing
            self.condprob = log(self.condprob + 1);
            % divide it by total number of words by class + 1
            % but it becomes subtracton in log
            for i = 1:2
                % add the size of vocabulary rather than 1
                self.condprob(i,:) = self.condprob(i,:)...
                    - log(totalcount(i)+size(X,2));
            end
            
        end
        
        % Predict with new data
        function predicted_classes = predict(self,new_data)
            % initialize variables
            X = new_data;
            predicted_classes = zeros(size(X,1),1);
            for i = 1:size(X,1)
                % get the indices of the non-zero features
                cols = X(i,:) ~= 0;
                % compute the joint probabilities by class
                % by summing log conditional probabilities
                scores = sum(self.condprob(:,cols),2);
                % convert prior to log and then add
                scores = scores + log(self.prior);
                % find the class with higher posterior probability
                [~,idx] = max(scores);
                if idx == 1
                    predicted_classes(i) = 1;
                else
                    predicted_classes(i) = 0;
                end
            end        
        end
    end
    
end

