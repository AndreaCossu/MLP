function [ eta , phialpha ] = ArmijoWolfeLS( phi0 , phip0 , nn , X , Y)

%
% Code adapted from Prof. Antonio Frangioni's implementation @ University of Pisa - Computer Science Department
%
%


% performs an Armijo-Wolfe Line Search.
%
% phi0 = phi( 0 ), phip0 = phi'( 0 ) < 0
%
% as > 0 is the first value to be tested: if phi'( as ) < 0 then as is
% divided by tau < 1 (hence it is increased) until this does not happen
% any longer
%
% m1 and m2 are the standard Armijo-Wolfe parameters; note that the strong
% Wolfe condition is used
%
% returns the optimal step and the optimal f-value
    
    as = 0.01;
    m1 = 0.0001;
    if nn.cg_flag
        m2 = 0.1;
    else
        m2 = 0.9;
    end
    tau = 0.5;
    max_feval = 1000;
    mina = 1e-16;
    sfgrd = 0.01;
    

    lsiter1 = 1;  % count iterations of first phase
    while (lsiter1-1) <= max_feval
       [ phialpha , phipalpha ] = f2phi( as , nn , X , Y );
       if ( phialpha <= phi0 + m1 * as * phip0 ) && ( abs( phipalpha ) <= - m2 * phip0 )
          eta = as;
          return;  % Armijo + strong Wolfe satisfied, we are done
       end
       if phipalpha >= 0
          break;
       end
       as = as / tau;
       lsiter1 = lsiter1 + 1;
    end    

    lsiter2 = 1;  % count iterations of second phase

    am = 0;
    eta = as;
    phipm = phip0;
    while ( (lsiter2+lsiter1-1) <= max_feval ) && ( ( as - am ) ) > mina && ( phipalpha > 1e-12 )
           % compute the new value by safeguarded quadratic interpolation
           eta = ( am * phipalpha - as * phipm ) / ( phipalpha - phipm );
           eta = max( [ am * ( 1 + sfgrd ) min( [ as * ( 1 - sfgrd ) eta ] ) ] );

           % compute phi( a )
           [ phialpha , phip ] = f2phi( eta , nn , X , Y);

           if ( phialpha <= phi0 + m1 * eta * phip0 ) && ( abs( phip ) <= - m2 * phip0 )
              break;  % Armijo + strong Wolfe satisfied, we are done
           end

           % restrict the interval based on sign of the derivative in a
           if phip < 0
              am = eta;
              phipm = phip;
           else
              as = eta;
              if as <= mina
                 break;
              end
              phipalpha = phip;
           end
           lsiter2 = lsiter2 + 1;
    end
end

function [ phi , phip ] = f2phi( alpha , nn , X , Y )
% phi( alpha ) = f( x - alpha * g )
% phi'( alpha ) = < \nabla f( x - alpha * g ) , g >

% returns phi(alpha), phi'(alpha)

    old_weights = nn.weights;
    old_grad = nn.grads;
    
    curr_dir = [];
    next_gradient = [];
    for i=1:nn.num_hidden+1
        if nn.cg_flag == 1 % conjugate gradient
            curr_dir = vertcat(curr_dir,nn.current_cg_dir{i}(:)); %take current direction
            nn.weights{i,1} = nn.weights{i,1} + (alpha .* nn.current_cg_dir{i,1}); % make a step in the current direction
        elseif nn.cg_flag == 0 % gradient descent
            curr_dir = vertcat(curr_dir,-nn.grads{i}(:)); 
            nn.weights{i,1} = nn.weights{i,1} - (alpha .* nn.grads{i,1});
        else
            error('In f2phi: wrong algorithm parameter.');
        end
    end
    % compute error and gradient for new weights
    [~, ~, phi] = nn.FP(X, Y);
    nn.BP(X,Y); 
    for i=1:nn.num_hidden+1
        % account for number of patterns
        nn.grads{i,1} = nn.grads{i,1} ./ nn.mb_size;
        next_gradient = vertcat(next_gradient,nn.grads{i}(:));
    end

    phip = next_gradient' * curr_dir;
    
    % restore old gradient and weights
    nn.weights = old_weights;
    nn.grads = old_grad;
end
