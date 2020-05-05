function [payoff]=PayoffExOption(ST)

payoff=max(ST(:,1)-ST(:,2),zeros(length(ST),1));

% for i=1:length(ST)
%     if ST(i,1)>ST(i,2)
%         payoff(i)=ST(i,1)-ST(i,2);
%     else
%         payoff(i)=0; 
%     end
% end