function [DoAs] = reorderDoAs(DoAs_old, DoAs_new, N_positions)
    % N_positions indicates the max number of positions between the
    % DoA predicted and the real one of a vehicle wrt to the other 
    % DoAs of other vehicles.
    % 
    % eg: N_positions = 3
    % eg: Real V1   V2   V3   V4   V5   V6
    %          DoA1 DoA2 DoA3 DoA4 DoA5 DoA6 
    % eg: Pred V1   V2   V3   V4   V5   V6
    %          DoA1 DoA2 DoA5 DoA3 DoA4 DoA6   

    
    % If the number of vehicle to track is less or equal then N_positions
    if N_positions >= size(DoAs_new, 2)
        
        N_positions = size(DoAs_new, 2);
        perm = perms(1:N_positions);
        
        % min_value, index_of_min
        trace = [10^10, 0];

        for p = 1 : size(perm, 1)
                
            diff = DoAs_old - DoAs_new(:,perm(p,:));
            
            diff_squared = diff.^2;
            
            minimum = min(diff_squared,[], 'all');
            
            if (minimum < trace(1))

                trace(1) = minimum;
                
                trace(2) = p;
                
            end
            
        end
        
        % Reorder correctly DoAs
        DoAs = DoAs_new(:,perm(trace(2),:));
        

    % If the number of vehicle to track is greater then N_positions    
    else
        % Initialization
        DoAs = DoAs_new;
        perm = perms(1:N_positions);
        
        for i = 1 : size(DoAs_new, 2) - N_positions +1

            % min_value, index_of_min
            trace = [10^10, 0];

            for p = 1 : size(perm, 1)
                
                diff = DoAs_old(:,i:N_positions+i-1) - DoAs_new(:,perm(p,:)+i-1);

                diff_squared = diff.^2;

                minimum = min(diff_squared,[], 'all');       
                
                if (minimum < trace(1) )

                    trace(1) = minimum;

                    trace(2) = p;

                end           
             
            end
            
            % Reorder correctly DoAs
            DoAs(:, i:N_positions+i-1) = DoAs(:,perm(trace(2),:) +i-1);
            
        end
        
        
    end
        
        
end

