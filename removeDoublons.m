function [X,y] = removeDoublons(X_pre, y_pre)
%% If we wanna remove the doublons % TODO : measure how much it consumes in term of memory/time
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
end