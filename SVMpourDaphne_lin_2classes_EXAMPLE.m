clc
clear all
close all
% Load Fisher's iris data set.  This loads in
%   meas(N,4) - feature vectors, each 4 dimensional
%   species{150} - class names: 'versicolor', 'virginica', 'setosa'
load fisheriris
% Let's keep only two classes.
indices1 = find(strcmp(species,'setosa')); 
indices2 = find(strcmp(species, 'versicolor'));
% Name them class 1 and class 2.
y = [ones(length(indices1),1); 2*ones(length(indices2),1)];
% We will just use 2 feature dimensions, since it is easier to visualize.
X = meas([indices1;indices2],1:2);
% However, when we do that there is a chance that some points will be
% duplicated (since we are ignoring the other features).  If so, just keep
% the first point.
indicesToKeep = true(size(X,1),1);
for i=1:size(X,1) % See if we already have the ith point.
    if any((X(i,1)==X(1:i-1,1)) & (X(i,2)==X(1:i-1,2)))
        indicesToKeep(i) = false;   % Skip this point
    end
end
allFeatureVectors = X(indicesToKeep, :);
allClasses = y(indicesToKeep);
numTotal = size(allFeatureVectors,1);

% Plot the vectors.
figure, hold on;
myColors = ['r', 'g'];
for j=1:numTotal
    plot(allFeatureVectors(j,1),allFeatureVectors(j,2), ...
    'Color', myColors(allClasses(j)), 'Marker', '.');
end
% Train SVM classifer.
cl = fitcsvm(allFeatureVectors,allClasses, ...
'KernelFunction', 'linear', ... % 'rbf', 'linear', 'polynomial'
'BoxConstraint', 1, ... % Default is 1
'ClassNames', [1,2]);
% Predict scores over the grid
d = 0.02;
[x1Grid,x2Grid] = meshgrid(min(allFeatureVectors(:,1)):d:max(allFeatureVectors(:,1)), ...
min(allFeatureVectors(:,2)):d:max(allFeatureVectors(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];
[~,scores] = predict(cl,xGrid);
% Plot the data and the decision boundary
figure;
h(1:2) = gscatter(allFeatureVectors(:,1),allFeatureVectors(:,2),allClasses,'rb','.');
hold on
h(3) = plot(allFeatureVectors(cl.IsSupportVector,1),allFeatureVectors(cl.IsSupportVector,2),'ko');
contour(x1Grid,x2Grid,reshape(scores(:,2),size(x1Grid)),[0 0],'k'); hold off