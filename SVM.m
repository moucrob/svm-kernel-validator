%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% http://cs-courses.mines.edu/csci508/schedule/18/Classification.pdf %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear all
tic
% close all
%% PARALLEL COMPUTING, OR NOT.
parallel = 0; 
%for 3 classes and ~100 measures, parallel takes 34sec while without takes just 5.
%%
%% How many times we sample the training set
nbSampling = 10;
%%
%% Preferences
myColors = ['r', 'g', 'b'];
myMarkers = ['o', 's', 'd'];
colors = 'rgb';
markers = 'osd';
alpha = 0.5; % Transparency of the 3D planes
%%
%% Loading of the Fisher's iris data set.  
%This loads in meas(N,4)
%- feature vectors, each 4 dimensional species{150} 
%- class names: 'versicolor', 'virginica', 'setosa'
load fisheriris
%% The 2 following code blocks have to be removed if our data loaded are already ready-to-use.
%% Build y the array of label of classes
indices1 = find(strcmp(species,'setosa')); 
indices2 = find(strcmp(species,'versicolor'));
indices3 = find(strcmp(species,'virginica')); 
%Labelize these classes with values 1, 2, 3, ...
y = [ones(length(indices1),1); 2*ones(length(indices2),1); 3*ones(length(indices2),1)];
%% Wheter we want visualize in 2D or 3D :
%% We will just use 2 feature dimensions, since it is easier to visualize.
X = meas([indices1;indices2;indices3],1:3);
% X = meas([indices1;indices2;indices3],2:3);
%HERE WE CHOOSE TO VISUALIZE 2D OR 3D AND WHICH COLUMN, IF REDUCED
%%
dim = size(X,2); % space of vizualisation
%% If we wanna remove the doublons % TODO : measure how much it consumes in term of memory/time
%Once one point is classified,
%"there is no point" to make again some effort for another same
[X,y] = removeDoublons(X,y);
%%
%%
nbrows = size(X,1); %number of measures
nbSpecies = length(unique(y));
%%
%% Divide the datas into ones for train and ones for test
%define the TUNABLE ratio between training datas and testing datas
ratio = 110/120; % %%%%%%%%%%%%%%%%%%%%%%%TUNABLE%%%%%%%%%%%%%%%%%%%%%%%%%
ratio = floor(ratio*nbrows); % bring it back to an integer
for k = 1:nbSampling
    [X_train, indexes] = datasample(X,ratio,1,'Replace',false); % false for never pick the same row
    y_train = y(indexes);
    idxCompl = 1:nbrows ; idxCompl(indexes) = [];
    X_test = X(idxCompl);
    y_test = y(idxCompl);
    %%
    %% Train SVM classifier for multiple classes, for the initial X_train.
    % Mdl = fitcecoc(X_train,y_train); %resubLoss(Mdl) = 6.5574% for the whole 122 datas
    % t = templateSVM('Standardize',1);
    % Mdl = fitcecoc(X,y,'Learners',t); %resubLoss(Mdl) = 4.918% for the whole 122 datas...
    %-> don't really know how adding 'Learners' improves somehow the Mdl...
    % Mdl.CodingMatrix % shows all the different one class vs another with +-1
    if parallel == 1
        delete(gcp('nocreate')) % if previous parallel computing session wasn't deleted
        pool = parpool; %for parallel computing (useful when many classes)
        options = statset('UseParallel',true);
    else
        options = statset('UseParallel',false);
    end
    % t = templateSVM('Standardize',1); %standardize the VM seems to decrease 
    %the misclassifying... but provides to draw accurate boundary lines, idkw...
    % Mdl = fitcecoc(X_train,y_train,'Learners',t,'Options',options,'Coding','onevsone');
    Mdl = fitcecoc(X_train,y_train,'Options',options,'Coding','onevsone');
    stock{k} = Mdl;
end
%%
%% Trustwhorthiness of the SOFT linear classifier, given the initial X_train
%Compute the in-sample classification error
disp(['(In the model due to this training set, ',num2str(resubLoss(Mdl)*100),'% of the measurements are outside of the predictional domain because of the compliance of the boundary lines.)'])
%resubLoss only after training, kfoldLoss only after cross validated.
%%
%% Hard training of the classifier (cross validation)
%All this block is a complete loss of time, doesn't make the following work.
% rng(1); % For reproducibility
% CVMdl = crossval(Mdl); %way to explore, as it summarize what i have done,
%if we want only the result the previous loop isn't necessary. But i made
%it to see the improvement into the line regressions proposed.
%Here crossval makes also 10 datasamplings by default.
% stock{nbSampling+1} = CVMdl;
%%
%% Plot whole datas and see how the Model, from the training datas only, fits.
%(2D dumb way of visualizing (by classifying all points on the grid) avoided.)
d = 0.05; %%%%%%%%%%%%% TUNABLE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[x1Grid,x2Grid] = meshgrid(min(X(:,1)):d:max(X(:,1)), ...
                           min(X(:,2)):d:max(X(:,2)));
fig = figure;
if dim == 2
    xGrid = [x1Grid(:),x2Grid(:)];
%     [classes,scores] = predict(CVMdl,X); %DOESN'T WORK
%     [classes,scores] = kfoldPredict(CVMdl,X); %DOESN'T WORK
    [classes,scores] = predict(Mdl,xGrid);
    %Hence this process is wrong, since i'm gonna plot the 10th sampling
    %predicted areas, while i would want to see my overall mean boundary
    %lines to match with some overall mean predictive areas, which are
    %implotable as long as i won't be able to find some predict() function
    %that can handle both the class of the output of fitcecoc() function
    %and the one in output of the crossvalidation() function.
    %BUT actually such a function could be easily handmade. It only require
    %to project points onto model equation and see from each side it
    %belongs ; AND not forget to also compare with other boundary lines
    %above/below the one we project on.
    
    %draw the trame, just to validate my boundary lines
    for j=1:length(classes)
        plot(xGrid(j,1), xGrid(j,2), '.', 'Color', myColors(classes(j))); hold on
    end
    %end of the validation code part, it can be removed as a commentary
    %Hence this following line is wrong but at least only 2 lines appear,
    %instead of 3 for the "right" ones, although we would only want N-1
    %for N classes (so 2 for 3)...
    % contour(x1Grid,x2Grid,reshape(scores(:,2),size(x1Grid)),[0 0],'r+');
    % hold on % Hence the way of drawing of the guy from mines US is not accurate.
    % plot(X(Mdl.IsSupportVector,1),X(Mdl.IsSupportVector,2),'ko');
    gscatter(X(:,1),X(:,2),y,colors,markers); hold on
%%% TODO : add the error due to the softness of the boundaries into the legend
    % data1 = gscatter(X(:,1),X(:,2),y,'rgb','.');
    % legend(data1)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    axis([min(X(:,1)) max(X(:,1)) ...
          min(X(:,2)) max(X(:,2))]); hold on
elseif dim == 3 %contour and gscatter work for only 2 arguments, not 3.
    for idx = 1:nbrows
        plot3(X(idx,1), X(idx,2), X(idx,3), 'Color', myColors(y(idx)), 'Marker', myMarkers(y(idx)));
        hold on
    end
    grid minor
    axis([min(X(:,1)) max(X(:,1)) ...
          min(X(:,2)) max(X(:,2)) ...
          min(X(:,3)) max(X(:,3))]); hold on
end
%%
%% Other attempt for boundary hyperplans :
for k = 1:nbSampling+1
%     disp(['k = ',num2str(k)])
    if dim == 2
        x1axe=linspace(min(X(:,1)),max(X(:,1)),1000);
        y = @(coeffdir,yInters,xVec) coeffdir*xVec + yInters;
        if k ~= nbSampling+1
            i=1; j=1;
            while 1
                a = stock{k}.BinaryLearners{i,1}.Beta(1);
                b = stock{k}.BinaryLearners{i,1}.Beta(2);
                c = stock{k}.BinaryLearners{i,1}.Bias;
                coef(k,j) = -a/b;
                y0(k,j) = -c/b;
                plot(x1axe, y(coef(k,j),y0(k,j),x1axe), 'k') ; hold on

                %For this trick please see my observations and bet below
                if (nbSpecies - i) == 0 % There won't be onevsall anymore
                    break
                else % we wanna skype the classifier onevsall
                    i=i+2;
                    j = j+1;
                end
            end
        else % k = nbSampling+1
            coef = mean(coef,1) ; y0 = mean(y0,1);
            for j = 1:nbSpecies-1
                plot(x1axe,y(coef(j),y0(j),x1axe), 'm', 'LineWidth', 2)
                hold on
            end
        end
        %some debugging :
        % plot(x1axe, -(Mdl.BinaryLearners{1,1}.Beta(1)/Mdl.BinaryLearners{1,1}.Beta(2))*x1axe - Mdl.BinaryLearners{1,1}.Bias/Mdl.BinaryLearners{1,1}.Beta(2), 'm', 'LineWidth',1) ; hold on
        % plot(x1axe, -(Mdl.BinaryLearners{2,1}.Beta(1)/Mdl.BinaryLearners{2,1}.Beta(2))*x1axe - Mdl.BinaryLearners{2,1}.Bias/Mdl.BinaryLearners{2,1}.Beta(2), 'm', 'LineWidth',1) ; hold on
        %so into the loop, the lines are drawn from the bottom to the top,
        %or more precisely for the class 1 to the last.
        %I bet that first a line is 1 vs all, then 1 vs 2, then 2 vs all that
        %is 2 vs 3 actually.
            %DOESN'T WORK
            %let's try an hybrid one, just to see it it fits better :
            %let's take the coeff or 1vs2 + the y0 of 1vsAll
            %plot(x1axe, -(Mdl.BinaryLearners{2,1}.Beta(1)/Mdl.BinaryLearners{2,1}.Beta(2))*x1axe - Mdl.BinaryLearners{1,1}.Bias/Mdl.BinaryLearners{1,1}.Beta(2), 'b', 'LineWidth',1) ; hold on
    elseif dim == 3
        Zmat = @(coeffdirX,coeffdirY,zInters,x1,x2) coeffdirX*x1 + coeffdirY*x2 + zInters;
        if k ~= nbSampling+1
            i=1; j=1;
            while 1
%                 disp(['j = ',num2str(j)])
                a = stock{k}.BinaryLearners{i,1}.Beta(1);
                b = stock{k}.BinaryLearners{i,1}.Beta(2);
                c = stock{k}.BinaryLearners{i,1}.Beta(3);
                d = stock{k}.BinaryLearners{i,1}.Bias;
                coefX(k,j) = -a/c;
                coefY(k,j) = -b/c;
                z0(k,j) = -d/c;
                surf(x1Grid, x2Grid, Zmat(coefX(k,j),coefY(k,j),z0(k,j),x1Grid,x2Grid), ...
                'FaceAlpha',alpha, 'EdgeColor', 'none'); hold on
            
                % For this trick please see my observations and bet just above.
                if (nbSpecies - i) == 0 % There won't be onevsall anymore
                    break
                else % we wanna skip the classifier onevsall
                    i=i+2;
                    j = j+1;
                end
            end
        else % k = nbSampling+1
            coefX = mean(coefX,1) ; coefY = mean(coefY,1) ; z0 = mean(z0,1);
            for j = 1:nbSpecies-1
                surf(x1Grid, x2Grid, Zmat(coefX(j),coefY(j),z0(j),x1Grid,x2Grid), ...
                'FaceAlpha',alpha, 'FaceColor', [1 0 1], 'EdgeColor', 'none'); hold on            
            end
        end
    end %otherwise, we plot nothing.
end

disp(['(It took ', num2str(toc), ' sec to compute and draw.)'])
%%
%% Test wheter the domains are well established
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DISABLABLE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if dim == 2
    while 1
        fprintf('Please ask to which class belongs a measure by clicking on the graph. \n');
        pointTest = ginput(1);
        plot(pointTest(1),pointTest(2),'Color','k','Marker','+'); hold on
        disp(['This point belongs to the domain ',num2str(predict(Mdl,pointTest)),'.'])
        %SAME REMARK, have to find a predict() function able to handle
        %CVMdl !!!
    end
% ... so maybe those aren't real lines
% -> does predict() really projects the pointTest onto the real line or
%    does he only use the grid generated ?   
end
%%
%%
hold off
%%
%% End the parallel session.
if parallel == 1
    delete(gcp('nocreate'))
end