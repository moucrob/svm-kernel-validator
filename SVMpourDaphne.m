% % Try to see the behaviour of Matlab precoded function fitcsvm with the
% % L1QP solver, through two sets of 2D points from each side of the line
% % y = x. Some points of the two sets burst the line, so that i can see
% % whether the SVM classifier will be soft or strict.
% %% TESTS
% % google : generate random coordinates inside a convex polytope
% 
% clc
% 
% upperEdges = [[0 0];[5 5];[1 0];[5 4];[0 5]];
% lowerEdges = [[0 0];[5 5];[0 1];[4 5];[5 0]];
% upperPolygon = convhulln(upperEdges);
% lowerPolygon = convhulln(lowerEdges);
% 
% % % To see the domains
% % plot([upperEdges(upperPolygon(:,1),1),upperEdges(upperPolygon(:,2),1)]', ...
% %      [upperEdges(upperPolygon(:,1),2),upperEdges(upperPolygon(:,2),2)]')
% % hold on
% % plot([lowerEdges(lowerPolygon(:,1),1),lowerEdges(lowerPolygon(:,2),1)]', ...
% %      [lowerEdges(lowerPolygon(:,1),2),lowerEdges(lowerPolygon(:,2),2)]')
% 
% % Creation of triangle simplex to spawning random points in a uniform space
% % way
% rowsUedges = size(upperEdges,1) ; rowsLedges = size(lowerEdges,1);
% upperCenterEdges = mean(upperEdges,1); lowerCenterEdges = mean(lowerEdges,1);
% upperEdges(end+1,:) = upperCenterEdges;
% lowerEdges(end+1,:) = lowerCenterEdges;
% upperIndexes = [upperPolygon,rowsUedges*ones(rowsUedges,1)];
% lowerIndexes = [lowerPolygon,rowsLedges*ones(rowsLedges,1)];
% % % To see the lattice :
% plot([upperEdges(upperIndexes(:,1),1),upperEdges(upperIndexes(:,2),1),upperEdges(upperIndexes(:,3),1)]', ...
%      [upperEdges(upperIndexes(:,1),2),upperEdges(upperIndexes(:,2),2),upperEdges(upperIndexes(:,3),2)]')
% hold on
% plot([lowerEdges(lowerIndexes(:,1),1),lowerEdges(lowerIndexes(:,2),1),lowerEdges(lowerIndexes(:,3),1)]', ...
%      [lowerEdges(lowerIndexes(:,1),2),lowerEdges(lowerIndexes(:,2),2),lowerEdges(lowerIndexes(:,3),2)]')
% 
% % Compute the relative areas of each vertex
% upperV = zeros(1,rowsUedges); lowerV = zeros(1,rowsLedges);
% for i=1:rowsUedges
%     upperV(i) = abs(det(upperEdges(upperIndexes(i,1:2),:)-upperCenterEdges));
%     lowerV(i) = abs(det(lowerEdges(lowerIndexes(i,1:2),:)-lowerCenterEdges));
% end
% upperV = upperV/sum(upperV) ; lowerV = lowerV/sum(lowerV);
% 
% % Generate points :
% nbUpper = 120 ; nbLower = nbUpper;
% r1U = rand(nbUpper,1); r1L = rand(nbLower,1);
% [~,~,simpindU] = histcounts(r1U,cumsum([0,upperV]));
% [~,~,simpindL] = histcounts(r1L,cumsum([0,lowerV]));
% % New sampling :
% r1U = rand(nbUpper,1); r1L = rand(nbLower,1);
% setU = upperEdges(upperIndexes(simpindU,1),:).*r1U + upperEdges(upperIndexes(simpindU,2),:).*(1-r1U);
% setL = lowerEdges(lowerIndexes(simpindL,1),:).*r1L + lowerEdges(lowerIndexes(simpindL,2),:).*(1-r1L);
% r2U = sqrt(rand(nbUpper,1)); r2L = sqrt(rand(nbLower,1));
% setU = setU.*r2U + upperEdges(upperIndexes(simpindU,3),:).*(1-r2U);
% setL = setL.*r2L + lowerEdges(lowerIndexes(simpindL,3),:).*(1-r2L);
% % To see the points :
% figure
% plot(setU(:,1),setU(:,2),'b.'); hold on; plot(setL(:,1),setL(:,2),'r.')
% 
% Mdl = fitcsvm([setU;setL],[ones(size(setU,1),1);-1*ones(size(setL,1),1)])

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% http://cs-courses.mines.edu/csci508/schedule/18/Classification.pdf %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc
clear all
tic
%close all
% Load Fisher's iris data set.  This loads in
%   meas(N,4) - feature vectors, each 4 dimensional
%   species{150} - class names: 'versicolor', 'virginica', 'setosa'
load fisheriris
nbSpecies = length(unique(species));
myColors = ['r', 'g', 'b'];
myMarkers = ['o', 's', 'd'];
colors = 'rgb';
markers = 'osd';

% Get the indices of the classes.
indices1 = find(strcmp(species,'setosa')); 
indices2 = find(strcmp(species,'versicolor'));
indices3 = find(strcmp(species,'virginica')); 
% Labelize these classes with values 1, 2, 3, ...
y_pre = [ones(length(indices1),1); 2*ones(length(indices2),1); 3*ones(length(indices2),1)];

% We will just use 2 feature dimensions, since it is easier to visualize.
X_pre = meas([indices1;indices2;indices3],1:3); % 2:3); % HERE WE CHOOSE TO VISUALIZE 2D OR 3D
dim = size(X_pre,2); % space of vizualisation

%% If we wanna remove the doublons
% % Once one point is classified, there is no point to make again some
% % effort for another same
indicesToKeep = true(size(X_pre,1),1);
for i=1:size(X_pre,1) % See if we already have the ith point.
    if any((X_pre(i,1)==X_pre(1:i-1,1)) & (X_pre(i,2)==X_pre(1:i-1,2)))
        indicesToKeep(i) = false;   % Skip this point
    end
end
X = X_pre(indicesToKeep, :);
y = y_pre(indicesToKeep);
%%

X = X_pre;
y = y_pre;
nbrows = size(X,1); %number of measures

% % Plot the datas.
% figure, hold on;
% for j=1:nbrows
%     plot(X(j,1),X(j,2), ...
%     'Color', myColors(y(j)), 'Marker', '.');
% end

%%
% Divide the datas into ones for train and ones for test
% define the TUNABLE ratio between training datas and testing datas
ratio = 110/120; % %%%%%%%%%%%%%%%%%%%%%%%TUNABLE%%%%%%%%%%%%%%%%%%%%%%%%%
ratio = floor(ratio*nbrows); % bring it back to an integer
[X_train, idx] = datasample(X,ratio,1,'Replace',false); % false for never pick the same row
y_train = y(idx);
idxCompl = 1:nbrows ; idxCompl(idx) = [];
X_test = X(idxCompl);
y_test = y(idxCompl);
%%

% Train SVM classifier for multiple classes.
% Mdl = fitcecoc(X_train,y_train); %resubLoss(Mdl) = 6.5574% for the whole 122 datas
% t = templateSVM('Standardize',1);
% Mdl = fitcecoc(X,y,'Learners',t); %resubLoss(Mdl) = 4.918% for the whole 122 datas...
% % -> don't really know how adding 'Learners' improves somehow the Mdl...
% % Mdl.CodingMatrix shows all the different one class vs another with +-1
parallel = 0; %for 3 classes and ~100 measures, parallel takes 34sec while without takes just 5.
if parallel == 1
    delete(gcp('nocreate')) % if previous parallel computing session wasn't deleted
    pool = parpool; %for parallel computing (useful when many classes)
    options = statset('UseParallel',true);
else
    options = statset('UseParallel',false);
end
% t = templateSVM('Standardize',1); %standardize the VM seems to decrease the misclassifying... but provides to draw accurate boundary lines, idkw...
% Mdl = fitcecoc(X_train,y_train,'Learners',t,'Options',options,'Coding','onevsone');
Mdl = fitcecoc(X_train,y_train,'Options',options,'Coding','onevsone');

% trustwhorthiness of the SOFT linear classifier
% Compute the in-sample classification error
disp(['(',num2str(resubLoss(Mdl)*100),'% of the measurements are outside of the predictional domain because of the compliance of the boundary lines.)'])
% resubLoss only after training, kfoldLoss only after cross validated.

%%
% For a kindof dumb way of visualizing by classifying all points on the grid.
d = 0.05; %%%%%%%%%%%%% TUNABLE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if dim == 2
    [x1Grid,x2Grid] = meshgrid(min(X(:,1)):d:max(X(:,1)), ...
                               min(X(:,2)):d:max(X(:,2)));
    xGrid = [x1Grid(:),x2Grid(:)];
    [classes,scores] = predict(Mdl,xGrid);
elseif dim == 3
    [x1GridBASE,x2GridBASE] = meshgrid(min(X(:,1)):d:max(X(:,1)), ...
                                       min(X(:,2)):d:max(X(:,2)));
    [x1Grid,x2Grid,x3Grid] = meshgrid(min(X(:,1)):d:max(X(:,1)), ...
                                      min(X(:,2)):d:max(X(:,2)), ...
                                      min(X(:,3)):d:max(X(:,3)));
    xGrid = [x1Grid(:),x2Grid(:),x3Grid(:)];
    [~,scores] = predict(Mdl,xGrid);
end
% Plot the decision boundaries
fig = figure;
if dim == 2
    % draw the trame, just to validate my boundary lines
    for j=1:length(classes)
        plot(xGrid(j,1), xGrid(j,2), '.', 'Color', myColors(classes(j))); hold on
    end
    % end of the validation code part, it can be removed as a commentary
    % hence this following line is wrong but at least only 2 lines appear,
    % instead of 3 for the "right" ones, although we would only want N-1
    % for N classes (so 2 for 3)...
%     contour(x1Grid,x2Grid,reshape(scores(:,2),size(x1Grid)),[0 0],'r+'); hold on
    % plot(X(Mdl.IsSupportVector,1),X(Mdl.IsSupportVector,2),'ko');
    gscatter(X(:,1),X(:,2),y,colors,markers); hold on
    % % TODO : add the error due to the softness of the boundaries into the legend
    % data1 = gscatter(X(:,1),X(:,2),y,'rgb','.');
    % legend(data1)
    axis([min(X(:,1)) max(X(:,1)) ...
          min(X(:,2)) max(X(:,2))]); hold on
elseif dim == 3 %contour and gscatter work for only 2 arguments, not 3.
    [~,~,id] = unique(y);
    for idx = 1:nbrows
        plot3(X(idx,1), X(idx,2), X(idx,3), 'Color', myColors(y(idx)), 'Marker', myMarkers(y(idx))); hold on
    end
    grid minor
    axis([min(X(:,1)) max(X(:,1)) ...
          min(X(:,2)) max(X(:,2)) ...
          min(X(:,3)) max(X(:,3))]); hold on
end
%%

% other attempt for boundary lines :
if dim == 2
    x1axe=linspace(min(X(:,1)),max(X(:,1)),1000);
    i=1;
    while 1
        a = Mdl.BinaryLearners{i,1}.Beta(1);
        b = Mdl.BinaryLearners{i,1}.Beta(2);
        c = Mdl.BinaryLearners{i,1}.Bias;
        coeffdir = -a/b;
        y0 = -c/b;
        plot(x1axe, coeffdir*x1axe + y0, 'k') ; hold on
        
        % For this trick please see my observations and bet just later:
        if (nbSpecies - i) == 0 % There won't be onevsall anymore
            break
        else % we wanna skype the classifier onevsall
            i=i+2;
        end
    end
    % % some debugging :
    %plot(x1axe, -(Mdl.BinaryLearners{1,1}.Beta(1)/Mdl.BinaryLearners{1,1}.Beta(2))*x1axe - Mdl.BinaryLearners{1,1}.Bias/Mdl.BinaryLearners{1,1}.Beta(2), 'm', 'LineWidth',1) ; hold on
    %plot(x1axe, -(Mdl.BinaryLearners{2,1}.Beta(1)/Mdl.BinaryLearners{2,1}.Beta(2))*x1axe - Mdl.BinaryLearners{2,1}.Bias/Mdl.BinaryLearners{2,1}.Beta(2), 'm', 'LineWidth',1) ; hold on
    % % so into the loop, the lines are drawn from the bottom to the top,
    % % or more precisely for the class 1 to the last.
    % % i bet that first a line is 1 vs all, then 1 vs 2, then 2 vs all that
    % % is 2 vs 3 actually.
        % % DOESN'T WORK
        % % let's try an hybrid one, just to see it it fits better :
        % % let's take the coeff or 1vs2 + the y0 of 1vsAll
        %plot(x1axe, -(Mdl.BinaryLearners{2,1}.Beta(1)/Mdl.BinaryLearners{2,1}.Beta(2))*x1axe - Mdl.BinaryLearners{1,1}.Bias/Mdl.BinaryLearners{1,1}.Beta(2), 'b', 'LineWidth',1) ; hold on

elseif dim == 3
    x1axe=linspace(min(X(:,1)),max(X(:,1)),1000);
    i=1;
    while 1
        a = Mdl.BinaryLearners{i,1}.Beta(1);
        b = Mdl.BinaryLearners{i,1}.Beta(2);
        c = Mdl.BinaryLearners{i,1}.Beta(3);
        d = Mdl.BinaryLearners{i,1}.Bias;
        coeffdirX = -a/c;
        coeffdirY = -b/c;
        z0 = -d/c;
        Zmat = @(x1,x2) coeffdirX*x1 + coeffdirY*x2 + z0;
        plane = surf(x1GridBASE, x2GridBASE, Zmat(x1GridBASE,x2GridBASE), 'FaceAlpha',0.9)
        plane.EdgeColor = 'none' ; hold on
        
        % For this trick please see my observations and bet just later:
        if (nbSpecies - i) == 0 % There won't be onevsall anymore
            break
        else % we wanna skype the classifier onevsall
            i=i+2;
        end
    end
else
    ; % we plot nothing
end

disp(['(It took ', num2str(toc), ' sec to compute and draw.)'])

%% Test wheter the domains are well established
if dim == 2
    continu = 'continue';
    while continu == 'continue'
        fprintf('Please ask to which class belongs a measure by clicking on the graph. \n');
        pointTest = ginput(1);
        plot(pointTest(1),pointTest(2),'Color','k','Marker','+'); hold on
        disp(['This point belongs to the domain ',num2str(predict(Mdl,pointTest)),'.'])
    end
% ... so maybe those aren't real lines 
% -> where is the accurate equation given by the SVM classifier?
% -> does predict() really projects the pointTest onto the real line or
%    does he only use the grid generated ?
    
end
%%

hold off

%Mdl.BinaryLearners{1}.Beta

% hard training of the classifier (cross validation)
% CVMdl = crossval(Mdl);

if parallel == 1
    delete(gcp('nocreate'))
end