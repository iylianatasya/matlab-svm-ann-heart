%Pseudocode%
%1. Load Heart Disease 
%2. Specify inputs and targets 
%3. Use cvpartition to define training and testing dataset 
%4. Create an SVM template. Standardize the predictors
%5. Train the ECOC classifier using the SVM template
%6. Compute ROC curve using kfoldPredict
%7. Compute confusion metrics using confusionchart
%8. Calculate average confusion matrix using accuracy formula 

load HeartDiseaseDataset.csv %load HeartDiseaseDataset

inputs = HeartDiseaseDataset(:,1:10); % Specify the row and columns for inputs and targets 
targets = HeartDiseaseDataset (:,11);

X = inputs
Y = targets

rng("default") % For reproducibility of the partition
%c = cvpartition(Y,Holdout=0.20);
c = cvpartition(Y,Holdout=0.30);
trainingIndices = training(c); % Indices for the training set
testIndices = test(c); % Indices for the test set
XTrain = X(trainingIndices,:);
YTrain = Y(trainingIndices);
XTest = X(testIndices,:);
YTest = Y(testIndices);
Mdl = fitcecoc(XTrain,YTrain);
% svm modeling 

t = templateSVM('Standardize',true,'SaveSupportVectors',true);

predictorNames = {'inputs','targets'};
%responseName = 'DiseaseStage';
classNames = {'0','1','2','3','4'}; % Specify class order
Mdl = fitcecoc(XTrain,YTrain,'Learners',t,'ClassNames',classNames)

%% Display class names and the coding design matrix
Mdl.ClassNames
Mdl.CodingMatrix

%% Access properties of the SVMs using cell subscripting and dot notation

L = size(Mdl.CodingMatrix,1); % Number of SVMs
sv = cell(L,1); % Preallocate for support vector indices
for j = 1:L
    SVM = Mdl.BinaryLearners{j};
    sv{j} = SVM.SupportVectors;
    sv{j} = sv{j}.*SVM.Sigma + SVM.Mu;
end
%% Plotting 
figure
gscatter(XTest(:,1),XTest(:,2),YTest);
hold on
markers = {'ko','ro','bo','go','yo'}; % Should be of length L
for j = 1:L
    svs = sv{j};
    plot(svs(:,1),svs(:,2),markers{j},...
        'MarkerSize',10 + (j - 1)*3);
end
title('Heart Disease -- ECOC Support Vectors')
xlabel(predictorNames{1})
ylabel(predictorNames{2})
legend([classNames,{'Support vectors - SVM 1',...
    'Support vectors - SVM 2','Support vectors - SVM 3','Support vectors - SVM 4','Support vectors - SVM 5'}],...
    'Location','Best')
hold off

%% ROC Curve
rng("default") % For reproducibility of the partition
Mdl = fitcecoc(XTest,YTest,Crossval = "on");
[~,Scores] = kfoldPredict(Mdl);
size(Scores)

Mdl.ClassNames % Display class names 
rocObj = rocmetrics(YTest,Scores,Mdl.ClassNames);

idx = strcmp(rocObj.Metrics.ClassName,Mdl.ClassNames(1));
rocObj.Metrics(idx,:)

plot(rocObj,AverageROCType="micro")

%% Confusion Metrics 
Mdl = fitcecoc(XTest,YTest); % Use test inputs and targets 

predictedY = resubPredict(Mdl); 

cm = confusionchart(YTest,predictedY); % Use confusionchart to compute confusion metrics

cm.NormalizedValues

cm.Title = 'Heart Disease Classification Using fitcecoc'; % Title of the confusion metrics 

cm.RowSummary = 'row-normalized';    % Display the classified and missclassified percentage for rows and columns
cm.ColumnSummary = 'column-normalized';

%% Accuracy

% average confusion matrix 
predictions = predict(Mdl,XTest);
con = confusionmat(YTest,predictions);
Accuracy = 100*sum(diag(con)/sum(con(:)))