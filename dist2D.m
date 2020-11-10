function [distance] = dist2D(s,d)
distance = sqrt(sum((s(1,1:2)-d(1,1:2)).^2));
end

