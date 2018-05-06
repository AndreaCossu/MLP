classdef MLP < handle
    % Multi Layer Perceptron Neural Network 
    properties
        %%
        % hyperparameters
        hidden_dim = []; % row vector with dimension of each hidden layer
        iterations = 100; % maximum number of iterations
        eta = 0.2; % learning rate
        num_hidden = 0; % number of hidden layers
        inp_dim = 1;
        out_dim = 1;
        num_patterns = 1; % size of dataset (number of patterns)
        lambda = 0.2; % Tikhonov hyperparameter
        alpha = 0.5; % Momentum hyperparameter
        reg = 0; % regularization on/off (1/0)
        bias = 0; % bias present or not present (1/0)
        threshold_grad = 1e-2; % stopping condition on norm of gradient
        mb_size = 0; % mini batch size
        
        % data structures for FP and BP
        weights = cell(1); % cell array of weight matrixes (one for each matrix)
        grads = cell(1); % gradient for each arc 
        momentum = cell(1); % deltaw old for the momentum
        deltas = cell(1); % delta for each unit
        nets = cell(1); % net for each unit
        outs = cell(1); % out for each unit
        
        % activation functions
        haf = @htanh; % activation function
        dhaf = @dhtanh; % derivative of haf
        oaf = @oaflogistic; % output activation function
        doaf = @doaflogistic; % derivative of oaf
        errorfunct = @MSE; % error function 
        derrorfunct = @dMSE; % derivative of errorfunct
        
        % accuracy and errors during FP
        accuracy = zeros(1,1); % accuracy vector (one element for each iteration) for tr set
        errors = zeros(1,1); % errors vector for tr set
        accuracy_val = zeros(1,1); % accuracy on validation set
        errors_val = zeros(1,1); % error on validation set
        
        % [left_range, right_range] is the range of initial random weights
        left_range = -0.7;
        right_range = 0.7;
        
        % conjugate gradient flag: 1 active, 0 not active
        cg_flag = 0;
        current_cg_dir = 0;
        previous_cg_dir = 0;
        previous_grads = 0;
        beta_type = "HS+";
        
        % line search flag parameters
        line_search = 0;
        
        % Neural Network used for regression (1) or for classification (0)
        regression = 0;

        % random seed
        seed = 0;

    end
    
    methods
        
    %%%-------------------------------------------------------%%%
        
        function nn = MLP(use,inp_dim,out_dim,hidden_dim,iterations,eta,lambda,alpha,bias,threshold,mb_size,cg_flag,beta_type,line_search)
            %%
            % use: 1 for regression, 0 for classification
            % inp_dim: dimension of input (one-of-k if needed)
            % out_dim: output dimension
            % hidden_dim: dimensionality of each hidden layer (vector)
            % iterations: maximum number of iterations
            % eta: learning rate
            % lambda: Tykhonov hyperparameter
            % alpha: momentum hyperparameter
            % bias: 1/0 - present/not present
            % threshold: threshold on gradient norm for stopping condition
            % mb_size: mini batch size
            % cg_flag: 1 to choose conjugate gradient, 0 to choose standard
            % backpropagation
            % line_search: 1 if line search has to be used, 0 otherwise.

            
            % check parameters correctness
            if mb_size < 1
                error('Mini batch size has to be greater or equal than 1');
            end
            if threshold < 0
                error('threshold on gradient must be greater than 0');
            end
            if lambda < 0 
                error('lambda must be between greater than 0');
            end
            if alpha < 0 || alpha > 1
                error('alpha must be between 0 and 1');
            end
            if size(hidden_dim,2) < 1 || size(hidden_dim,1) ~= 1
                error('hidden_dim must be a row vector');
            end
            if sum(arrayfun( @(x) x<=0 , hidden_dim)) > 0
                error('each element of hidden_dim must be greater than 0');
            end
            if iterations < 1
                error('iterations must be greater than 0');
            end
            if eta < 0
                error('eta must be greater than 0');
            end
            if bias ~= 0 && bias ~= 1
                error('bias must be 1 or 0');
            end
            if inp_dim <= 0 || out_dim <= 0
                error('inp_dim and out_dim have to be greater than 0');
            end
            if lambda > 0
                nn.reg = 1;
            else
                nn.reg = 0;
            end
            if use~=0 && use~=1
                error('use must be 1 or 0');
            end
            if line_search ~=0 && line_search ~= 1
                error('line search must be 1 or 0');
            end
            % if cg_flag is provided check it and assign it
            if nargin > 11
                if cg_flag~=0 && cg_flag~=1
                    error('cg_flag must be 1 or 0');
                else
                    nn.cg_flag = cg_flag;
                end   
            end
            % if beta_type is provided set it
            if nargin > 12
                if beta_type ~= "HS+" && beta_type ~= "PR+" && beta_type ~= "FR"
                    error('Beta allowed: HS+, PR+, FR');
                else
                    nn.beta_type = beta_type;
                end
            end
            if nargin > 13
                nn.line_search = line_search;
            end
                        
            % regression or classification
            nn.regression = use;
            
            if nn.regression
                nn.oaf = @oid;
                nn.doaf = @doid;
                nn.errorfunct = @MSE;
                nn.derrorfunct = @dMSE;
            end
            
            
            % assign parameters values
            nn.hidden_dim = hidden_dim;
            nn.eta = eta;
            nn.alpha = alpha;
            nn.lambda = lambda;
            nn.iterations = iterations;
            nn.num_hidden = size(nn.hidden_dim,2);
            nn.bias = bias;
            nn.inp_dim = inp_dim;
            nn.out_dim = out_dim;
            nn.threshold_grad = threshold;
            nn.mb_size = mb_size;
            
            % data structures init
            nn.weights = cell(nn.num_hidden+1,1);
            nn.grads = cell(nn.num_hidden+1,1);
            nn.deltas = cell(nn.num_hidden+1,1);
            nn.nets = cell(nn.num_hidden+1,1);
            nn.outs = cell(nn.num_hidden+1,1);
            nn.momentum = cell(nn.num_hidden+1,1);
            
            nn.accuracy = zeros(1,nn.iterations);
            nn.errors = zeros(1,nn.iterations);
            nn.accuracy_val = zeros(1,nn.iterations);
            nn.errors_val = zeros(1,nn.iterations);
            
            nn.current_cg_dir = cell(nn.num_hidden+1,1);
            nn.previous_cg_dir = cell(nn.num_hidden+1,1);
            nn.previous_grads = cell(nn.num_hidden+1,1);

            nn.seed = 1234;
            rng(nn.seed); % set the seed for random number generation
            
        end
    %%%-------------------------------------------------------%%%
    
        
        function [out,correct,error] = FP(nn,x,y) 
            % REQUIRE
            % x : data matrix with one pattern per column
            % y (optional) : target with one target per column
            %
            % RETURN
            % out: matrix output of the network, one output per column
            % correct: number of correctly classified patterns
            % error: errors on x
            % IGNORE CORRECT IF Regression
                     
            % get net and output for first hidden layer
            nn.nets{1,1} = nn.weights{1,1}*x;
            nn.outs{1,1} = nn.haf(nn.nets{1,1});
            if nn.bias
                % add a constant 1 unit to the layer
                nn.outs{1,1}(size(nn.outs{1,1},1)+1,:) = 1;
            end
            
            % repeat for other hidden layers and output
            for i=2:(nn.num_hidden+1)
                nn.nets{i,1} = nn.weights{i,1}*nn.outs{i-1,1};              
                % if on hidden layer use haf
                if i<nn.num_hidden+1
                    nn.outs{i,1} = nn.haf(nn.nets{i,1});
                    if nn.bias
                        nn.outs{i,1}(size(nn.outs{i,1},1)+1,:) = 1;
                    end
                % if on output layer use oaf and do not add bias
                else
                    nn.outs{i,1} = nn.oaf(nn.nets{i,1}); 
                end
            end  
            
            % assign output of the network as the output of last layer
            out = nn.outs{nn.num_hidden+1,1};
            
            % if target is provided compute if output is correct or not
            if nargin > 2
                if ~nn.regression
                    correct = sum(round(out) == y);
                else
                    correct = NaN;
                end
                error = nn.errorfunct(y,nn);
            else
                correct = NaN;
                error = NaN;
            end
        end
        
    %%%-------------------------------------------------------%%%

        
    function [] = BP(nn,out0,target)
        %%
        % REQUIRE
        % out0: output of input layer (the actual input). Matrix with one
        % pattern per column
        % target: matrix with one target per column
        
        
        % calc deltas
        % output units
        nn.deltas{nn.num_hidden+1,1} = nn.derrorfunct(target,nn.outs{nn.num_hidden+1,1}) .* (nn.doaf(nn.nets{nn.num_hidden+1,1}));    
        % hidden units (starting from last layer)
        for i=nn.num_hidden:-1:1
            if ~nn.bias
                sumk = nn.weights{i+1,1}' * nn.deltas{i+1,1};
            else
                % bias units are not included in the sum over k of deltak * w_kj
                sumk = (nn.weights{i+1,1}(:,1:end-1))' * nn.deltas{i+1,1};
            end
            nn.deltas{i,1} = sumk .* nn.dhaf(nn.nets{i,1});
        end

        % gradient sum
        % first matrix
        nn.grads{1,1} = nn.deltas{1,1} * out0';
        
        % other matrixes
        for m=2:nn.num_hidden+1
            nn.grads{m,1} = nn.deltas{m,1} * nn.outs{m-1,1}';
        end  
    end
        
    %%%-------------------------------------------------------%%%
    
        function [acc,acc_val,errs,errs_val,iterations] = fit(nn,X,y,val_data,val_target)
            % REQUIRE
            % X : data matrix with one pattern per row
            % y : target matrix with one target per row
            % val_data (optional) : data matrix for validation set (only used to plot
            % during training) - same format as X
            % val_target (optional) : target matrix for validation set (only used to
            % plot during training) - same format as y
            
            % RETURN
            % acc : (vector of length iterations) accuracy on training set
            % acc_val : (vector of length iterations) accuracy on validation set
            % errs : (vector of length iterations) errors on training set
            % errs_val : (vector of length iterations) errors on validation set
            % iterations : how many iterations done
            % IGNORE ACC, ACC_VAL IF Regression
            
            if size(X,1) < 1 || size(X,2) < 1 || size(y,1) < 1 || size(y,2) < 1
                error('Input patterns and target patterns have incorrect dimensions');
            end
            
            
            iterations = 0;
            nn.num_patterns = size(X,1);
            
            %% INITIALIZATION OF DATA STRUCTURES
            % column cell array of weight matrixes
            % NOTE: bias units have no input connections
            nn.weights{1,1} = (nn.right_range - nn.left_range) * rand(nn.hidden_dim(1,1),nn.inp_dim+nn.bias) + nn.left_range;	%pesi random tra -0.7 e 0.7 per il primo hl
            if nn.num_hidden > 1
              for i=2:nn.num_hidden
                nn.weights{i,1} = (nn.right_range - nn.left_range) * rand(nn.hidden_dim(1,i),nn.hidden_dim(1,i-1)+nn.bias) + nn.left_range;
              end
            end
            nn.weights{nn.num_hidden+1,1} = (nn.right_range - nn.left_range) * rand(nn.out_dim,nn.hidden_dim(1,nn.num_hidden)+nn.bias) + nn.left_range;
            
            % momentum
            for i=1:nn.num_hidden+1
                nn.momentum{i,1} = zeros(size(nn.weights{i,1}));
            end            
            
            % set an initial previous_cg_direction
            for i=1:nn.num_hidden+1
                nn.previous_cg_dir{i,1} = zeros(size(nn.weights{i,1})); 
            end
            
            for i=1:nn.num_hidden+1
                nn.grads{i,1} = zeros(size(nn.weights{i,1}));
            end
            
            % calculate number of total weights
            nweights = 0;
            for i=1:nn.num_hidden+1
                nweights = nweights + size(nn.weights{i,1},1)*size(nn.weights{i,1},2);
            end
            

            % main loop - epochs            
            for iter=1:nn.iterations
                %%
                % Stopping condition: ||grad|| < threshold_grad
                
                if iter > 1
                    sumgrad = 0;
                    for i=1:nn.num_hidden+1
                        sumgrad = sumgrad + norm(nn.grads{i,1},'fro');
                    end
                    if sumgrad < nn.threshold_grad
                        break
                    end
                end             
                
                iterations = iterations + 1;
                
                % random permutation of data
                idx = randperm(nn.num_patterns);
                X_rand = X(idx,:);
                y_rand = y(idx,:);
                
                left_index = 1-nn.mb_size; % the first pattern in the current minibatch
                right_index = 0; % the last pattern in the current minibatch
                tot_corr = 0; % number of misclassification on this iteration
                tot_err = 0;         
                
                % loop minibatch
                % for every batch (discard last patterns if not divisible
                % per minibatch size
                while (right_index + nn.mb_size) <= nn.num_patterns
                    %%
                    
                    % update current minibatch positions
                    left_index = left_index + nn.mb_size;
                    right_index = right_index + nn.mb_size;
                    
                    % take minibatch data
                    X_batch = X_rand(left_index:right_index,:);
                    y_batch = y_rand(left_index:right_index,:);
                    
                    %%
                    % Forward Propagation
                    out0 = X_batch'; % column vector
                    target = y_batch'; % row vector

                    % account for bias
                    if nn.bias
                        out0(size(out0,1)+1,:) = 1;
                    end

                    % accuracy summing
                    [~,corr,err] = nn.FP(out0,target);
                    if ~nn.regression
                        tot_corr = tot_corr + corr;
                    end
                    tot_err = tot_err + err;
                    
                    nn.previous_grads = nn.grads;
                    
                    % Back Propagation
                    nn.BP(out0,target);
                    
                    % account for number of patterns at the end of each
                    % minibatch
                    for i=1:nn.num_hidden+1
                        nn.grads{i,1} = nn.grads{i,1} ./ nn.mb_size;
                    end

                    % delta rule at the end of each minibatch
                    if ~nn.cg_flag
                        % standard delta rule
                        for m=1:nn.num_hidden+1
                            regularizer = nn.lambda .* nn.weights{m,1};
                            if nn.line_search
                                % compute in advance the length of vector g
                                totsize = 0;
                                for i=1:nn.num_hidden+1
                                    totsize = totsize + (size(nn.grads{i},1)*size(nn.grads{i},2));
                                end
                                % fill the vector g and make it a single dimensional vector
                                g = zeros(totsize,1);
                                start = 1;
                                for i=1:nn.num_hidden+1
                                    len = (size(nn.grads{i},1)*size(nn.grads{i},2));
                                    g(start:start-1+len,1) = nn.grads{i}(:);
                                    start = start + len;
                                end
                                % calculate phip0
                                derr = -g' * g;;
                                [step, ~] = ArmijoWolfeLS(err, derr, nn, out0, target);

                                gradient_descent = (-1 * step) .* nn.grads{m,1};
                            else 
                                gradient_descent = (-1 * nn.eta) .* nn.grads{m,1};
                            end
                            nn.momentum{m,1} = nn.alpha .* nn.momentum{m,1} + gradient_descent;
                            nn.weights{m,1} = nn.weights{m,1} + nn.momentum{m,1} - regularizer;  
                        end
                    else
                        % conjugate gradient
                        nn.cg_direction(iter,nweights);
                        if nn.line_search                            
                            % compute in advance the length of vector g
                            totsize = 0;
                            for i=1:nn.num_hidden+1
                                totsize = totsize + (size(nn.grads{i},1)*size(nn.grads{i},2));
                            end
                            % fill the vector g and dir
                            g = zeros(totsize,1);
                            dir = zeros(totsize,1);
                            % transform g and dir in a single dimension vector
                            start = 1;
                            for i=1:nn.num_hidden+1
                                len = (size(nn.grads{i},1)*size(nn.grads{i},2));
                                g(start:start-1+len,1) = nn.grads{i}(:);
                                dir(start:start-1+len,1) = nn.current_cg_dir{i}(:);
                                start = start + len;
                            end
                            % calculate phip0
                            derr = dir' * g;
                            [step, ~] = ArmijoWolfeLS(err, derr, nn, out0, target);
                        end
                        for m=1:nn.num_hidden+1
                            regularizer = nn.lambda .* nn.weights{m,1};
                            if nn.line_search
                                nn.weights{m,1} = nn.weights{m,1} + (step .* nn.current_cg_dir{m,1}) - regularizer;
                            else
                                nn.weights{m,1} = nn.weights{m,1} + (nn.eta .* nn.current_cg_dir{m,1}) - regularizer;
                            end
                        end
                        nn.previous_cg_dir = nn.current_cg_dir;
                    end

                end % end of main iteration (all pattern in dataset learned)
                
                
                if ~nn.regression
                    nn.accuracy(1,iter) = tot_corr / right_index;
                end
                nn.errors(1,iter) = tot_err / right_index;               
                
                % calculate statistics on validation after a full batch
                if nargin > 3
                    [acc_val,~,err_val] = nn.test(val_data,val_target);
                    if ~nn.regression
                        nn.accuracy_val(1,iter) = acc_val;
                    end
                    nn.errors_val(1,iter) = err_val;
                end

            end % end of main loop - all epochs done
            
            % return accuracy values
            if ~nn.regression
                acc = nn.accuracy;
            else
                acc = NaN;
            end
            errs = nn.errors;
            
            if nargin > 3
                if ~nn.regression
                    acc_val = nn.accuracy_val;
                else
                    acc_val = NaN;
                end
                errs_val = nn.errors_val;
            else
                acc_val = NaN;
                errs_val = NaN;
            end
        end
        
    %%%--------------------------------------------------------------------------------%%%    
    
        function [accuracy,results,errors] = test(nn,X,y)
            %%
            % REQUIRE
            % X: data matrix (one pattern per row, one feature per column)
            % y (optional): target matrix (one target per row, one feature per column)
            % 
            
            % RETURN
            % accuracy : accuracy on the test set
            % results : matrix with the results of the network (one result
            % per column)
            % errors : errors on test set
            % IGNORE ACCURACY IF Regression
            
            num_test = size(X,1); % number of test patterns
                        
            % input
            out0=X'; % each pattern on column
            % account for bias
            if nn.bias
                out0(size(out0,1)+1,:) = 1;
            end
            
            if nargin > 2
                target = y'; % row vector

                % Forward Propagation
                [output,correct,errs] = nn.FP(out0,target);

                if ~nn.regression
                    accuracy = correct / num_test;
                else
                    accuracy = NaN;
                end
                errors = errs / num_test;
            else
                [output,~,~] = nn.FP(out0);  
                accuracy = NaN;
                errors = NaN;
            end
            results = output;
            
        end
        
        %%%--------------------------------------------------------------------------------%%%
        
        function [] = cg_direction(nn,iter,nweights)
            %% CONJUGATE GRADIENT 
            % iter = current iteration number of optimization algorithm
            % nweights = number of weights of neural network - used to
            % restart direction
            
            if iter == 1
                % return the opposite of the gradient if first iteration
                for i=1:nn.num_hidden+1
                    nn.current_cg_dir{i,1} = -1 .* nn.grads{i,1};
                end
            elseif rem(iter,nweights)==0 && nn.beta_type=="FR"
                % restart only for FR every n iterations
                % HS+,PR+ are restarted automatically
                for i=1:nn.num_hidden+1
                    nn.previous_cg_dir{i,1} = zeros(size(nn.weights{i,1}));
                    nn.current_cg_dir{i,1} = -1 .* nn.grads{i,1};
                end
            else
                % beta: Hestenes-Stiefel (HS), with Gilbert and Nocedal fix
                % to assure global convergence (HS+) 
                beta = cell(nn.num_hidden+1,1);
                y = cell(nn.num_hidden+1,1);
                % on the other iterations
                % take the opposite of the gradient and add a multiple of
                % previous direction
                for i=1:nn.num_hidden+1
                    y{i,1} = nn.grads{i,1} - nn.previous_grads{i,1};
                    grads_reshaped = nn.grads{i,1}(:);
                    y_reshaped = y{i,1}(:);
                    previous_cg_dir_reshaped = nn.previous_cg_dir{i,1}(:);
                    switch nn.beta_type
                        case "HS+"
                            % beta must be a scalar
                            % reshape read one columns at time and append it below
                            % the previous read column

                            % reshape grads and y in column vector to obtain a
                            % scalar
                            % beta{i,1} = (nn.grads{i,1}' * y{i,1}) / 
                            %             (y{i,1}' * nn.previous_cg_dir{i,1});
                            beta{i,1} = max(0, (grads_reshaped' * y_reshaped) / (y_reshaped' * previous_cg_dir_reshaped));                    
                        case "FR"
                            % Fletcher-Reeves beta
                            % ||grad||^2 / ||prev_grad||^2
                            beta{i,1} = (norm(nn.grads{i,1},'fro')^2) / (norm(nn.previous_grads{i,1},'fro')^2);
                        case "PR+"
                            % Polak-Ribière beta
                            % g_k * y_k-1 / ||g_k-1||^2
                            beta{i,1} = max(0,(grads_reshaped' * y_reshaped) / norm(nn.previous_grads{i,1},'fro')^2);
                    end
                    
                    % use the standard conjugate gradient direction
                    nn.current_cg_dir{i,1} = (-1 .* nn.grads{i,1}) + beta{i,1} .* nn.previous_cg_dir{i,1};                    
                end
            end
        end
    end
end
