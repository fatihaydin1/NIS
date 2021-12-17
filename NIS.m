%% Nimble Instance Selection (NIS)
function [ idx ] = NIS( X, varargin )

    narginchk(1, 2);
    
    if nargin == 2
        alpha = varargin{1};
    else
        alpha = 1;
    end

    transformedX = round(alpha*((X-min(X,[],1,'omitnan'))./(std(X,'omitnan'))),0);
    transformedX(isnan(transformedX)) = 0;
    
    [~, idx, ~] = unique(transformedX,'rows', 'first');
end
