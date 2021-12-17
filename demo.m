%% Demo for the NIS algorithm
clc;
clear;

% Select a data set
dataset = load('nomao');
fns = fieldnames(dataset);
[ X, Y ] = divideTable( dataset.(fns{1}) );

alpha = 0.3;
[ACC, R, T] = demo1(X, Y, 'KNN', alpha);

[ idx, newX, newY ] = demo2( X, Y, alpha );

clear dataset;
clear fns;



%%
function [ACC, R, T] = demo1( X, Y, classifier, alpha )

    predictions = repmat(Y, 1, 2);
    indices = crossvalind('Kfold', Y, 10);
    R = zeros(10,1);
    T = zeros(10,1);
        
    for i = 1:10
        fprintf('%d',i);
        test = (indices == i);
        train = ~test;
                
        trainY = Y(train,:);
        trainX = X(train,:);
        testX = X(test,:);
        
        tic;
        idx = NIS(trainX, alpha);
        T(i) = toc;
        newTrainX = trainX(idx, :);
        newTrainY = trainY(idx);

        switch classifier
            case 'KNN'
                Mdl = fitcknn(newTrainX, newTrainY, 'NumNeighbors', 1);
            case 'CART'
                Mdl = fitctree(newTrainX, newTrainY);
                predictions(test, 2) = predict(Mdl, testX);
            case 'NB'
                % "normal", "mn", "kernel", "mvmn".
                Mdl = fitcnb(newTrainX, newTrainY, 'DistributionNames', 'normal');
                predictions(test, 2) = predict(Mdl, testX);
            case 'SVM'
                % "linear", "gaussian", "rbf", "polynomial"
                t = templateSVM('Standardize', true, 'KernelFunction', 'linear');
                Mdl = fitcecoc(newTrainX, newTrainY, 'Learners', t);
                predictions(test, 2) = predict(Mdl, testX);
        end
        R(i) = size(newTrainX,1);
        predictions(test, 2) = predict(Mdl, testX);
    end
    ACC = sum(predictions(:,1) == predictions(:,2))*100/length(Y);
    R = 100 - (mean(R)*100/length(Y));
    T = sum(T);
end



%%
function [ idx, newX, newY ] = demo2( X, Y, alpha )

    idx = NIS(X, alpha);
    newX = X(idx, :);
    newY = Y(idx);
end



%% Separate the dataset into the input matrix and the output vector
function [ X, Y ] = divideTable( DATASET )

    if istable(DATASET)
        X = table2array(DATASET(:,1:end-1));        
        Y = categorical(DATASET.Class);
    else
        error('The parameter must be a table, not a %s.', class(DATASET));
    end
end