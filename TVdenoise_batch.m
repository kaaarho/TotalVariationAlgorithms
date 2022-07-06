clc; clear;
% Denoising/smoothing a given image y with the isotropic total variation.
%
% The iterative algorithm converges to the unique image x minimizing 
%
% ||x-y||_2^2/2 + lambda.TV(x)
%
% TV(x)=||Dx||_1,2, where D maps an image to its gradient field.
%
% The over-relaxed Chambolle-Pock algorithm is described in
% L. Condat, "A primal-dual splitting method for convex optimization 
% involving Lipschitzian, proximable and linear composite terms", 
% J. Optimization Theory and Applications, vol. 158, no. 2, 
% pp. 460-479, 2013.
%
% Code written by Laurent Condat, CNRS research fellow in the
% Dept. of Images and Signals of GIPSA-lab, Univ. Grenoble Alpes, 
% Grenoble, France.
%
% Version 1.1, Oct. 12, 2016


load('./data/imgs.mat')
load('./GaussianDistribution2DMask_30.mat')
mask = double(maskRS2);

Nbiter= 400;	% number of iterations
lambda = 0.005; 	% regularization parameter ori0.1
tau = 0.05;		% proximal parameter >0; influences the convergence speed

savedir = './data/result/';

for i=1:2000
    
fprintf('%d/2000\n',i)
img=squeeze(img_ori(i,:,:));

x_good  = squeeze(img_ori(i,:,:)); 

y = fftshift(fft2(x_good)); 
y = (double(y)) .* (mask);
x_bad = ifft2(ifftshift(y));
x_bad = abs(x_bad);

x_generated = TVdenoising(x_bad,lambda,tau,Nbiter);

%% Save Image
imwrite(abs(x_good),[savedir, 'groundtruth/groungtruth_',int2str(i),'.png'])
imwrite(abs(x_generated),[savedir, 'generated/generated_',int2str(i),'.png'])
imwrite(abs(x_bad),[savedir,'bad/bad_',int2str(i),'.png'])

end



function x = TVdenoising(y,lambda,tau,Nbiter)
	
	rho = 1.98;		% relaxation parameter, in [1,2)
	sigma = 1/tau/8; % proximal parameter
	[H,W]=size(y);

	opD = @(x) cat(3,[diff(x,1,1);zeros(1,W)],[diff(x,1,2) zeros(H,1)]);
	opDadj = @(u) -[u(1,:,1);diff(u(:,:,1),1,1)]-[u(:,1,2) diff(u(:,:,2),1,2)];	
	prox_tau_f = @(x) (x+tau*y)/(1+tau);
	prox_sigma_g_conj = @(u) bsxfun(@rdivide,u,max(sqrt(sum(u.^2,3))/lambda,1));
	
	x2 = y; 		% Initialization of the solution
	u2 = prox_sigma_g_conj(opD(x2));	% Initialization of the dual solution
	cy = sum(sum(y.^2))/2;
	primalcostlowerbound = 0;
		
	for iter = 1:Nbiter
		x = prox_tau_f(x2-tau*opDadj(u2));
		u = prox_sigma_g_conj(u2+sigma*opD(2*x-x2));
		x2 = x2+rho*(x-x2);
		u2 = u2+rho*(u-u2);
		if mod(iter,25)==0
			primalcost = norm(x-y,'fro')^2/2+lambda*sum(sum(sqrt(sum(opD(x).^2,3))));
			dualcost = cy-sum(sum((y-opDadj(u)).^2))/2;
				% best value of dualcost computed so far:
			primalcostlowerbound = max(primalcostlowerbound,dualcost);
				% The gap between primalcost and primalcostlowerbound is even better
				% than between primalcost and dualcost to monitor convergence. 
			fprintf('nb iter:%4d  %f  %f  %e\n',iter,primalcost,...
				primalcostlowerbound,primalcost-primalcostlowerbound);
% 			figure(3);
% 			imshow(x);
		end
	end
end
