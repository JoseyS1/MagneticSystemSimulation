function matHeiswolff1_GPU_Functions_Bloced()

tic
%#ok<*NASGU>
clear
clear samRand
rng(314,'simdTwister')


LMin =5;
LMax = 5;
Lsz = 1;



LtMin = 10;
LtMax = 15;
Ltsz = 10;

NCONF  = 1;
NEQ    = 50;
NMESS  = 50;     % Monte Carlo sweeps
TMIN   = 1.0;
TMAX   = 2.0;
DT     = -0.2;      % temperature control
NTEMP  = ceil(1+(TMIN-TMAX)/DT);

impconc = 0.0;%.407253;                          %% impurities as decimal

Flat_Dist = true ;
Tail_Heavy_Dist = false; % Distribution of site coupling strength if both are set to true the tail heavy distribution is always used
MIN = 1;   % Minimum value for flat distribution
MAX = 1;   % Maximum value for flat distribution
y_exponent = 0.0; % Exponent value for tail heavy distribution



% number of disorder configs
CANON_DIS = false;
PRINT_ALL = false;                        % print data about individual sweeps
WRITE_ALL = false;                      % write individual configs




NCLUSTER0 = 10;                          % initial cluster flips per sweep




%mx;my;mz;sweepmag;sweepen                  % magnetization vector
%sweepmagqt;sweepmagq2; sweepmagq3;magqt;magqs     % mag(qtime);mag(qspace);
%Gt;Gs;Gtcon;Gscon                         % G(qspace); G(qtime); connected versions
%mag; mag2; mag4; bin; susc;en; en2;sph     % magnetization; its square; energy
%xit;xis;xitcon;xiscon                      % correlation lengths in space an time; connected versions
%glxit;glxis;glxitcon;glxiscon              % global correlation lengths in space an time; connected versions

temperature  = zeros(NTEMP,NCONF);
conf_mag     = zeros(NTEMP,NCONF);
conf_mag2   = zeros(NTEMP,NCONF);
conf_2mag   = zeros(NTEMP,NCONF);
conf_mag4   = zeros(NTEMP,NCONF);              % configuration averages
conf_logmag = zeros(NTEMP,NCONF);
conf_susc    = zeros(NTEMP,NCONF);
conf_bin      = zeros(NTEMP,NCONF);
conf_2susc  = zeros(NTEMP,NCONF);
conf_2bin    = zeros(NTEMP,NCONF);              % conf. av. of squared observables
conf_en      = zeros(NTEMP,NCONF);
conf_sph     = zeros(NTEMP,NCONF);
conf_2en     = zeros(NTEMP,NCONF);
conf_2sph    = zeros(NTEMP,NCONF);
conf_Gt       = zeros(NTEMP,NCONF);
conf_Gs       = zeros(NTEMP,NCONF);
conf_Gtcon  = zeros(NTEMP,NCONF);
conf_Gscon  = zeros(NTEMP,NCONF);
conf_xit       = zeros(NTEMP,NCONF);
conf_xis       = zeros(NTEMP,NCONF);
conf_xitcon  = zeros(NTEMP,NCONF);
conf_xiscon  = zeros(NTEMP,NCONF);







cell_temperature = cell(LMax,LtMax);
cell_conf_mag    = cell(LMax,LtMax);
cell_conf_mag2   = cell(LMax,LtMax);
cell_conf_2mag   = cell(LMax,LtMax);
cell_conf_mag4   = cell(LMax,LtMax);              % configuration averages
cell_conf_logmag = cell(LMax,LtMax);
cell_conf_susc   = cell(LMax,LtMax);
cell_conf_bin    = cell(LMax,LtMax);
cell_conf_2susc  = cell(LMax,LtMax);
cell_conf_2bin   = cell(LMax,LtMax);              % conf. av. of squared observables
cell_conf_en     = cell(LMax,LtMax);
cell_conf_sph    = cell(LMax,LtMax);
cell_conf_2en    = cell(LMax,LtMax);
cell_conf_2sph   = cell(LMax,LtMax);
cell_conf_Gt     = cell(LMax,LtMax);
cell_conf_Gs     = cell(LMax,LtMax);
cell_conf_Gtcon  = cell(LMax,LtMax);
cell_conf_Gscon  = cell(LMax,LtMax);
cell_conf_xit    = cell(LMax,LtMax);
cell_conf_xis    = cell(LMax,LtMax);
cell_conf_xitcon = cell(LMax,LtMax);
cell_conf_xiscon = cell(LMax,LtMax);



cTmag = zeros(NEQ,NTEMP);
cTen = zeros(NEQ,NTEMP);
cell_cTmag = cell(LMax,LtMax);
cell_cTen = cell(LMax,LtMax);

% cTmag2 = zeros(NEQ,NTEMP);
% cTmag4 = zeros(NEQ,NTEMP);
tt1 = 0;
tt2 = 0;


measT = 0;
eqT = 0;

for L = LMin:Lsz:LMax
    for Lt = LtMin:Ltsz:LtMax
        checkerboard = generateCheckerboard(L,Lt);
        checkerboardOther = ~checkerboard;
        L2 = L*L;
        L3 = Lt*L2;         % L=linear system size
        qspace = 2*pi/L;
        qtime = 2*pi/Lt;   % minimum q values for correlation length
        
        %parfor
        for k = 1:NCONF
            fprintf('L = %d , Lt = %d: Disorder configuration %d of %d \n',L,Lt,k,NCONF)
            
            % Reset spins
            %sBlock = zeros(L,L,Lt,3,'single','gpuArray');
            
            

            
            
            
            % Initialize site couplings
            %[j1,j3,j5] = initializeSiteCouplings(L,Lt,L2,L3,Flat_Dist,Tail_Heavy_Dist,MIN,MAX,y_exponent);
            [j1p,j1m,j2p,j2m,j3p,j3m] = initializeSiteCouplings_Block(L,Lt,Flat_Dist,Tail_Heavy_Dist,MIN,MAX,y_exponent);
            
            
            % Initialize impurities
            occu = initializeSiteImpurities(L,Lt,impconc);
            % Initialize spins
            is = 1:L3;
            
            hs = getRSphere_WholeArray(L,Lt);
            
            sBlock = bsxfun(@times,occu,hs);
            
            
            
            occs = gpuArray(logical(occu));
            
            
            
            
            %hold on
            % Loop over temperatures
            ncluster = NCLUSTER0;
            T = TMAX;
            %itemp = 1;
            for itemp = 1:NTEMP%while T > TMIN
                beta= 1/T;
                
                %fprintf('T = %d \n',T)
                
                
                % Equilibration, carry out NEQ full MC steps
%                 nm = 1;
%                 tic
                %tic
                %for ii = 1:10
%                 [sBlock] = metro_sweep_faster_Combined_allGPU(sBlock,L,Lt,beta,occu,j1p,j1m,j2p,j2m,j3p,j3m,checkerboard,checkerboardOther,NEQ);
%                 for ii = 1:NEQ/10
%                     [sBlock] = metro_sweep_faster_Combined_allGPU(sBlock,L,Lt,beta,occu,j1p,j1m,j2p,j2m,j3p,j3m,checkerboard,checkerboardOther,NEQ/10);
% %                     [sBlock] = metro_sweep_faster(sBlock,L,Lt,beta,occu,j1p,j1m,j2p,j2m,j3p,j3m,checkerboard)
% %                     [sBlock] = metro_sweep_faster(sBlock,L,Lt,beta,occu,j1p,j1m,j2p,j2m,j3p,j3m,checkerboardOther)
% %                     
%                 end

                
                    [sBlock] = metro_sweep_faster_Combined_allGPU(sBlock,L,Lt,beta,occu,j1p,j1m,j2p,j2m,j3p,j3m,checkerboard,checkerboardOther,NEQ);
%                     for is = 1:1
%                         [sBlock] = wolff_sweep_notGPU(NCLUSTER0,sBlock,L,Lt,beta,occu,j1p,j1m,j2p,j2m,j3p,j3m);
%                     end
%                 eqT = eqT + toc;
%                 tt1 = tt1+toc;
%                 tic
%                 [sBlock] = metro_sweep_faster_Combined(sBlock,L,Lt,beta,occu,j1p,j1m,j2p,j2m,j3p,j3m,checkerboard,checkerboardOther,NEQ);
%                 tt2 = tt2+toc;
%                 for isweep=1:NEQ
%                     if nm <= 20
%                         T0 = T+(1)^(nm+1)*DT/nm;
%                         beta = 1/T0;
%                         beta = 1/T;
%                     else
%                         beta = 1/T;
%                     end
%                     %wolff_sweep(nsweepflip);
%                     [sBlock] = metro_sweep_faster(sBlock,L,Lt,beta,occu,j1p,j1m,j2p,j2m,j3p,j3m,checkerboard);
%                     
%                     [sBlock] = metro_sweep_faster(sBlock,L,Lt,beta,occu,j1p,j1m,j2p,j2m,j3p,j3m,checkerboardOther);
%                     
% %                     [sBlock] = metro_sweep_faster_Combined(sBlock,L,Lt,beta,occu,j1p,j1m,j2p,j2m,j3p,j3m,checkerboard,checkerboardOther);
%                     
%                     %[sBlock] = wolff_sweep(NCLUSTER0,sBlock,L,Lt,beta,occu,j1p,j1m,j2p,j2m,j3p,j3m,checkerboard);
%                     if NCLUSTER0 >0
%                         [sBlock] = wolff_sweep_notGPU(NCLUSTER0,sBlock,L,Lt,beta,occu,j1p,j1m,j2p,j2m,j3p,j3m);
%                     end
%                     %[cTen(isweep,itemp),~,cTmag(isweep,itemp),~,~] = measurement_faster(sBlock,L,Lt,j1p,j2p,j3p);
%                     nm = nm+1;
%                 end

                %plot(cTen(:,itemp))
                %plot(cTmag(:,itemp))
                %pause(.1)
                totalflip=0;
                %         for isweep=1:NEQ/2
                %             %metro_sweep();
                %             [sBlock] = metro_sweep_faster(sBlock,L,Lt,beta,occu,j1p,j1m,j2p,j2m,j3p,j3m,checkerboard);
                %             [sBlock] = metro_sweep_faster(sBlock,L,Lt,beta,occu,j1p,j1m,j2p,j2m,j3p,j3m,checkerboardOther);
                %             %wolff_sweep(nsweepflip);
                %             %totalflip = totalflip + nsweepflip;
                %         end
                avclsize = totalflip/ncluster/(NEQ/2);
                ncluster = L3/avclsize + 0.5 + 2;
                
                % Measuring loop, carry out NMESS full MC sweeps
                
            mag = 0.0;
            mag2 = 0.0;
            mag4 = 0.0;
            en =  0.0;
            en2= 0.0;
            magqt = 0.0;
            magqs = 0.0;
            Gt =  0.0;
            Gs =  0.0;
               % tic
                totalflip = 0;
%                 for isweep = 1:NMESS
%                     [en_inc,en2_inc,mag_inc,mag2_inc,mag4_inc] = measurement_faster(sBlock,L,Lt,j1p,j2p,j3p);
%                     
%                     
%                     [sBlock] = metro_sweep_faster_Combined_allGPU(sBlock,L,Lt,beta,occu,j1p,j1m,j2p,j2m,j3p,j3m,checkerboard,checkerboardOther,1);
% %                      [sBlock] = metro_sweep_faster(sBlock,L,Lt,beta,occu,j1p,j1m,j2p,j2m,j3p,j3m,checkerboard);
% %                      [sBlock] = metro_sweep_faster(sBlock,L,Lt,beta,occu,j1p,j1m,j2p,j2m,j3p,j3m,checkerboardOther);
%                     if NCLUSTER0>0
%                         [sBlock] = wolff_sweep_notGPU(NCLUSTER0,sBlock,L,Lt,beta,occu,j1p,j1m,j2p,j2m,j3p,j3m);
%                     end
%                     %wolff_sweep(nsweepflip);
%                     %totalflip=totalflip+nsweepflip;
% 
%                     
%                     en   = en   + (en_inc   - en)/isweep;
%                     en2  = en2  + (en2_inc  - en2)/isweep;
%                     mag  = mag  + (mag_inc  - mag)/isweep;
%                     mag2 = mag2 + (mag2_inc - mag2)/isweep;
%                     mag4 = mag4 + (mag4_inc - mag4)/isweep;
% %                     aa(isweep) = mag2_inc;
% %                     bb(isweep) = mag4_inc;
% %                     aa(isweep) = mag2;
% %                     bb(isweep) = mag4;
% %                     sprintf('%d %d',mag2,mag4)
%                     %corr_func();
%                 end


                [en,en2,mag,mag2,mag4] = measurement_faster_combinedMetro(sBlock,L,Lt,beta,occu,j1p,j1m,j2p,j2m,j3p,j3m,checkerboard,checkerboardOther,NMESS);

                %plot(aa)
%                 hold on
%                 plot(bb)
%                 drawnow
%                 hold on
                
%                 measT = measT + toc;
                
                magqt = magqt / NMESS;
                magqs = magqs / NMESS;
                Gt    = Gt / NMESS;
                Gs    = Gs / NMESS;
                
                susc    = (mag2 - mag^2)*L3*beta;
                bin     = 1 - mag4/(3*mag2^2);
                sph     = (en2 - en^2)*L3*beta^2;
                Gtcon   = Gt - magqt^2;
                Gscon   = Gs - magqs^2;
                
                
                
                xit    = (mag2 - Gt)/ (Gt*qtime*qtime);
                xit    = sqrt(abs(xit));
                xis    = (mag2 - Gs)/ (Gs*qspace*qspace);
                xis    = sqrt(abs(xis));
                xitcon = ((mag2 - mag^2) - Gtcon)/ (Gtcon*qtime*qtime);
                xitcon = sqrt(abs(xitcon));
                xiscon = ((mag2 - mag^2) - Gscon)/ (Gscon*qspace*qspace);
                xiscon = sqrt(abs(xiscon));
                
                
                % Collect Data
                conf_mag(itemp,k)    =  mag;
                conf_2mag(itemp,k)   =  mag^2;
                conf_mag2(itemp,k)   =  mag2;
                conf_mag4(itemp,k)   =  mag4;
                conf_logmag(itemp,k) =  log(mag);
                conf_susc(itemp,k)   =  susc;
                conf_bin(itemp,k)    =  bin;
                conf_2susc(itemp,k)  =  susc^2;
                conf_2bin(itemp,k)   =  bin^2;
                conf_en(itemp,k)     =  en;
                conf_sph(itemp,k)    =  sph;
                conf_2en(itemp,k)    =  en^2;
                conf_2sph(itemp,k)   =  sph^2;
                conf_Gt(itemp,k)     =  Gt;
                conf_Gs(itemp,k)     =  Gs;
                conf_Gtcon(itemp,k)  =  Gtcon;
                conf_Gscon(itemp,k)  =  Gscon;
                conf_xit(itemp,k)    =  xit;
                conf_xis(itemp,k)    =  xis;
                conf_xitcon(itemp,k) =  xitcon;
                conf_xiscon(itemp,k) =  xiscon;
                
                
                
                

                
                
                
                avclsize = (totalflip/ncluster)/NMESS;
                ncluster = L3/avclsize + 0.5 + 2;
                
                temperature(itemp,k) = T;
                %itemp    = itemp + 1;
                T        = T + DT;
            end % Temperature
            
            %plot(temperature(:,k),1-conf_mag4(:,k)./(3*conf_mag2(:,k).^2))
            
        end
        
        
        
                cell_temperature {L,Lt} = temperature;
                cell_conf_mag    {L,Lt} = conf_mag   ;
                cell_conf_mag2   {L,Lt} = conf_mag2  ;
                cell_conf_2mag   {L,Lt} = conf_2mag  ;
                cell_conf_mag4   {L,Lt} = conf_mag4  ;              % configuration averages
                cell_conf_logmag {L,Lt} = conf_logmag;
                cell_conf_susc   {L,Lt} = conf_susc  ;
                cell_conf_bin    {L,Lt} = conf_bin   ;
                cell_conf_2susc  {L,Lt} = conf_2susc ;
                cell_conf_2bin   {L,Lt} = conf_2bin  ;              % conf. av. of squared observables
                cell_conf_en     {L,Lt} = conf_en    ;
                cell_conf_sph    {L,Lt} = conf_sph   ;
                cell_conf_2en    {L,Lt} = conf_2en   ;
                cell_conf_2sph   {L,Lt} = conf_2sph  ;
                cell_conf_Gt     {L,Lt} = conf_Gt    ;
                cell_conf_Gs     {L,Lt} = conf_Gs    ;
                cell_conf_Gtcon  {L,Lt} = conf_Gtcon ;
                cell_conf_Gscon  {L,Lt} = conf_Gscon ;
                cell_conf_xit    {L,Lt} = conf_xit   ;
                cell_conf_xis    {L,Lt} = conf_xis   ;
                cell_conf_xitcon {L,Lt} = conf_xitcon;
                cell_conf_xiscon {L,Lt} = conf_xiscon;
        
    end
end

hold on
%plot(temperature,conf_mag)
% plot(temperature,conf_mag2)
% plot(temperature,conf_mag4)
%plot(temperature,conf_en)
%plot(temperature,1-conf_mag4./(3*conf_mag2.^2))

if~exist('k','var')
    k = NCONF;
end
% errorbar(temperature(:,k),mean(1-conf_mag4(:,1:k)./(3*conf_mag2(:,1:k).^2),2),std(1-conf_mag4(:,1:k)./(3*conf_mag2(:,1:k).^2),0,2)/sqrt(k))




for L = LMin:Lsz:LMax
    for Lt = LtMin:Ltsz:LtMax
        temp = temperature(:,k);
        binder = mean(1-cell_conf_mag4{L,Lt}(:,1:k)./(3*cell_conf_mag2{L,Lt}(:,1:k).^2),2);
        err = std(1-cell_conf_mag4{L,Lt}(:,1:k)./(3*cell_conf_mag2{L,Lt}(:,1:k).^2),0,2)/sqrt(k);
        
        errorbar(temp, binder  ,err)
        
    end
end

toc
end





























function [sBlock] = metro_sweep_faster(sBlock,L,Lt,beta,occu,j1p,j1m,j2p,j2m,j3p,j3m,checkerboard)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    carries out one Metropolis sweep
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%         j1pe = repmat(j1p,1,1,1,3);
%         j2pe = repmat(j2p,1,1,1,3);
%         j3pe = repmat(j3p,1,1,1,3);
%         j1me = repmat(j1m,1,1,1,3);
%         j2me = repmat(j2m,1,1,1,3);
%         j3me = repmat(j3m,1,1,1,3);


%nSpins = getRSphere_GPU2(nP);
nSpins = getRSphere_WholeArray(L,Lt);

nSpins = bsxfun(@times,occu,nSpins);
ds = (sBlock - nSpins);

[s1p,s1m,s2p,s2m,s3p,s3m] = updateNeighborSpins(sBlock,L,Lt);

%         j1pr = repmat(j1p,1,1,1,3);
%         j1mr = repmat(j1m,1,1,1,3);
%         j2pr = repmat(j2p,1,1,1,3);
%         j2mr = repmat(j2m,1,1,1,3);
%         j3pr = repmat(j3p,1,1,1,3);
%         j3mr = repmat(j3m,1,1,1,3);
%
%         tic
%         for kk = 1:100
%             %netxyz = bsxfun(@times,j1p,s1p) + bsxfun(@times,j2p,s2p) + bsxfun(@times,j3p,s3p) + bsxfun(@times,j1m,s1m) + bsxfun(@times,j2m,s2m) + bsxfun(@times,j3m,s3m);
%             netxyz = j1pr.*s1p+ j2pr.*s2p + j3pr.*s3p+j1mr.*s1m + j2mr.*s2m+j3mr.*s3m;
%         end
%         toc
%
%                 tic
%         for kk = 1:100
netxyz = bsxfun(@times,j1p,s1p) + bsxfun(@times,j2p,s2p) + bsxfun(@times,j3p,s3p) + bsxfun(@times,j1m,s1m) + bsxfun(@times,j2m,s2m) + bsxfun(@times,j3m,s3m);
%            % netxyz = j1pr.*s1p+ j2pr.*s2p + j3pr.*s3p+j1mr.*s1m + j2mr.*s2m+j3mr.*s3m;
%         end
%         toc
%
%           tic
%         for kk = 1:100
%             %netxyz = bsxfun(@times,j1p,s1p) + bsxfun(@times,j2p,s2p) + bsxfun(@times,j3p,s3p) + bsxfun(@times,j1m,s1m) + bsxfun(@times,j2m,s2m) + bsxfun(@times,j3m,s3m);
%             netxyz = j1p.*s1p+ j2p.*s2p + j3p.*s3p+j1m.*s1m + j2m.*s2m+j3m.*s3m;
%         end
%         toc


%         cudaFilename = 'F:\JoseySync\PhysicsProjects\DisorderedSystems\Cuda\test3.cu';
%         ptxFilename = 'F:\JoseySync\PhysicsProjects\DisorderedSystems\Cuda\test3.ptx';
%         kernel = parallel.gpu.CUDAKernel( ptxFilename, cudaFilename );
%
%         kernel.ThreadBlockSize = [4*L,1,1];
%         kernel.GridSize = [L*Lt*3/4,1,1];
%
%
%
%         netxyzT = zeros(L,L,Lt,3,'single','gpuArray');
%
%                 tic
%         for kk = 1:100
%         netxyzT = feval(kernel, netxyzT,j1pr,j1mr,j2pr,j2mr,j3pr,j3mr,s1p,s1m,s2p,s2m,s3p,s3m);
%         end
%         toc

%         n1 = 250;
%         n2 = 100;
%         n3 = 3;
%
%         cudaFilename = 'F:\JoseySync\PhysicsProjects\DisorderedSystems\Cuda\example2.cu';
%         ptxFilename = 'F:\JoseySync\PhysicsProjects\DisorderedSystems\Cuda\example2.ptx';
%
%
%         oGPU = ones(n1,n1,n2,n3,'double','gpuArray');
%         rGPU = 1.1+rand(n1,n1,n2,n3,'double','gpuArray');
%
%
%         kernel = parallel.gpu.CUDAKernel( ptxFilename, cudaFilename,'add2');
%
%
%         kernel.GridSize = [n1*n2*3,1,1];
%
%
%         kernel.ThreadBlockSize = [n1,1,1];
%
%         oGPU = feval(kernel,oGPU,rGPU);
%         min(oGPU,[],'all')

%netxyzT = times(j1p,s1p) + times(j2p,s2p) + times(j3p,s3p) + times(j1m,s1m) + times(j2m,s2m) + times(j3m,s3m);
% netxyzT = arrayfun(@calculateNetxyz,s1p,s1m,s2p,s2m,s3p,s3m,j1p,j1m,j2p,j2m,j3p,j3m);




dE = sum(netxyz.*(ds),4);




swap = (dE<0 | (exp(-dE*beta) > rand(L,L,Lt,'single','gpuArray')))&checkerboard;



notswap = ~swap;


sBlock = bsxfun(@times,swap,nSpins) + bsxfun(@times,notswap,sBlock);


end





function [sBlock] = metro_sweep_faster_Combined(sBlock,L,Lt,beta,occu,j1p,j1m,j2p,j2m,j3p,j3m,checkerboard,checkerboardOther,NEQ)
for i = 1:NEQ
    
    nSpins = getRSphere_WholeArray(L,Lt);
    
    nSpins = bsxfun(@times,occu,nSpins);
    ds = (sBlock - nSpins);
    
    [s1p,s1m,s2p,s2m,s3p,s3m] = updateNeighborSpins(sBlock,L,Lt);
    
    
    netxyz = bsxfun(@times,j1p,s1p) + bsxfun(@times,j2p,s2p) + bsxfun(@times,j3p,s3p) + bsxfun(@times,j1m,s1m) + bsxfun(@times,j2m,s2m) + bsxfun(@times,j3m,s3m);
    
    
    
    dE = sum(netxyz.*(ds),4);
    
    
    
    
    swap = (dE<0 | (exp(-dE*beta) > rand(L,L,Lt,'single','gpuArray')))&checkerboard;
    
    
    
    notswap = ~swap;
    
    
    sBlock = bsxfun(@times,swap,nSpins) + bsxfun(@times,notswap,sBlock);
    
    
    
    %%%%%%%%%%%%%% Other checkerboard
    
    
    nSpins = getRSphere_WholeArray(L,Lt);
    
    nSpins = bsxfun(@times,occu,nSpins);
    ds = (sBlock - nSpins);
    
    [s1p,s1m,s2p,s2m,s3p,s3m] = updateNeighborSpins(sBlock,L,Lt);
    
    
    netxyz = bsxfun(@times,j1p,s1p) + bsxfun(@times,j2p,s2p) + bsxfun(@times,j3p,s3p) + bsxfun(@times,j1m,s1m) + bsxfun(@times,j2m,s2m) + bsxfun(@times,j3m,s3m);
    
    
    
    dE = sum(netxyz.*(ds),4);
    
    
    
    
    swap = (dE<0 | (exp(-dE*beta) > rand(L,L,Lt,'single','gpuArray')))&checkerboardOther;
    
    
    
    notswap = ~swap;
    
    
    sBlock = bsxfun(@times,swap,nSpins) + bsxfun(@times,notswap,sBlock);
end

end





function [sBlock] = metro_sweep_faster_Combined_allGPU(sBlock,L,Lt,beta,occu,j1p,j1m,j2p,j2m,j3p,j3m,checkerboard,checkerboardOther,NEQ)
occu = gpuArray(occu);
j1p               =  gpuArray(j1p);              
j1m               =  gpuArray(j1m);              
j2p               =  gpuArray(j2p);              
j2m               =  gpuArray(j2m);              
j3p               =  gpuArray(j3p);              
j3m               =  gpuArray(j3m);              
checkerboard      =  gpuArray(checkerboard);     
checkerboardOther =  gpuArray(checkerboardOther);

p1 = gpuArray(circshift(1:L,-1));
m1 = gpuArray(circshift(1:L,1));

tp1 = gpuArray(circshift(1:Lt,-1));
tm1 = gpuArray(circshift(1:Lt,1));


j1pr = repmat(j1p,1,1,1,3);
j2pr = repmat(j2p,1,1,1,3);
j3pr = repmat(j3p,1,1,1,3);

j1mr = repmat(j1m,1,1,1,3);
j2mr = repmat(j2m,1,1,1,3);
j3mr = repmat(j3m,1,1,1,3);



jrs = zeros(L,L,Lt,3,6,'single','gpuArray');

jrs(:,:,:,:,1) = j1pr;
jrs(:,:,:,:,2) = j2pr;
jrs(:,:,:,:,3) = j3pr;
jrs(:,:,:,:,4) = j1mr;
jrs(:,:,:,:,5) = j2mr;
jrs(:,:,:,:,6) = j3mr;



% sbt = zeros(L+2,L+2,Lt+2,3,'single','gpuArray');

neqStepSize = 25;
% nSpinsTest = getRSphere_WholeArray_N(L,Lt,2*neqStepSize);
% nSpinsTest = permute(nSpinsTest,[1 2 3 5 4]);
% nSpinsTest = bsxfun(@times,occu,nSpinsTest);
% 
% rands = samRand(L,L,Lt,2*neqStepSize);

j = 1;
for i = 1:NEQ
    if mod(i,neqStepSize)==1
        nSpinsTest = getRSphere_WholeArray_N(L,Lt,2*NEQ);
        nSpinsTest = permute(nSpinsTest,[1 2 3 5 4]);
        nSpinsTest = bsxfun(@times,occu,nSpinsTest);
        
        rands = samRand(L,L,Lt,2*NEQ);
        j = 1;
    end
    %nSpins = getRSphere_WholeArray(L,Lt);
    
    nSpins = nSpinsTest(:,:,:,:,j);


    %[s1p,s1m,s2p,s2m,s3p,s3m] = updateNeighborSpins(sBlock,L,Lt);

%     ss(:,:,:,:,1) = sBlock(p1,:,:,:);ss(:,:,:,:,2) = sBlock(:,p1,:,:);ss(:,:,:,:,3) = sBlock(:,:,p1,:);ss(:,:,:,:,4) = sBlock(m1,:,:,:);ss(:,:,:,:,5) = sBlock(:,m1,:,:);ss(:,:,:,:,6) = sBlock(:,:,m1,:);nxyzT = sum(ss.*jrs,5);
%     sbt(2:1+L,2:1+L,2:1+Lt,:) = sBlock;
    
    s1p = sBlock(p1,:,:,:);s1m = sBlock(m1,:,:,:);s2p = sBlock(:,p1,:,:);s2m = sBlock(:,m1,:,:);s3p = sBlock(:,:,tp1,:);s3m = sBlock(:,:,tm1,:);
%     netxyz = j1pr.*s1p + j2pr.*s2p + j3pr.*s3p + j1mr.*s1m + j2mr.*s2m + j3mr.*s3m;
    netxyz = sum(cat(5,s1p,s2p,s3p,s1m,s2m,s3m).*jrs,5);    
    


    ds = (sBlock - nSpins);
    dE = sum(netxyz.*(ds),4);
    

    
    swap = (dE<0 | (exp(-dE*beta) > rands(:,:,:,j)))&checkerboard;
    j=j+1;
    notswap = ~swap;
    sBlock = bsxfun(@times,swap,nSpins) + bsxfun(@times,notswap,sBlock);
    
    %%%%%%%%%%%%%% Other checkerboard
    %nSpins = getRSphere_WholeArray(L,Lt);
    nSpins = nSpinsTest(:,:,:,:,j);
    %nSpins = bsxfun(@times,occu,nSpins);
    ds = (sBlock - nSpins);
    s1p = sBlock(p1,:,:,:);s1m = sBlock(m1,:,:,:);s2p = sBlock(:,p1,:,:);s2m = sBlock(:,m1,:,:);s3p = sBlock(:,:,tp1,:);s3m = sBlock(:,:,tm1,:);
%     netxyz = j1pr.*s1p + j2pr.*s2p + j3pr.*s3p + j1mr.*s1m + j2mr.*s2m + j3mr.*s3m;
    netxyz = sum(cat(5,s1p,s2p,s3p,s1m,s2m,s3m).*jrs,5);   
    dE = sum(netxyz.*(ds),4);
    swap = (dE<0 | (exp(-dE*beta) > rands(:,:,:,j)))&checkerboardOther;
    
    j=j+1;
    notswap = ~swap;
    sBlock = bsxfun(@times,swap,nSpins) + bsxfun(@times,notswap,sBlock);
end

end









function [en,en2,mag,mag2,mag4] = measurement_faster(sBlock,L,Lt,j1p,j2p,j3p)

% measures energy and magnetization




mxGPU = sum(sBlock(:,:,:,1),'all');
myGPU = sum(sBlock(:,:,:,2),'all');
mzGPU = sum(sBlock(:,:,:,3),'all');

[s1p,~,s2p,~,s3p,~] = updateNeighborSpins(sBlock,L,Lt);

netxyz = bsxfun(@times,j1p,s1p) + bsxfun(@times,j2p,s2p) + bsxfun(@times,j3p,s3p);
sweepenGPU = sum(netxyz.*(sBlock),'all');


sweepen = gather(sweepenGPU);

en  =  sweepen/(L^2*Lt);  %  + sweepen/L3;
en2 = en^2; % + (sweepen/L3)^2;

mag =   gather(sqrt(mxGPU^2 + myGPU^2 + mzGPU^2))/(L^2*Lt);


%mag  = mag;%  +  sweepmag;
mag2 = mag^2;% +  sweepmag^2;
mag4 = mag^4;% +  sweepmag^4;
mag = mag;
end %subroutine measurement







function [en,en2,mag,mag2,mag4] = measurement_faster_combinedMetro(sBlock,L,Lt,beta,occu,j1p,j1m,j2p,j2m,j3p,j3m,checkerboard,checkerboardOther,NEQ)

% measures energy and magnetization
occu = gpuArray(occu);
j1p               =  gpuArray(j1p);              
j1m               =  gpuArray(j1m);              
j2p               =  gpuArray(j2p);              
j2m               =  gpuArray(j2m);              
j3p               =  gpuArray(j3p);              
j3m               =  gpuArray(j3m);              
checkerboard      =  gpuArray(checkerboard);     
checkerboardOther =  gpuArray(checkerboardOther);

p1 = gpuArray(circshift(1:L,-1));
m1 = gpuArray(circshift(1:L,1));

tp1 = gpuArray(circshift(1:Lt,-1));
tm1 = gpuArray(circshift(1:Lt,1));


j1pr = repmat(j1p,1,1,1,3);
j2pr = repmat(j2p,1,1,1,3);
j3pr = repmat(j3p,1,1,1,3);

j1mr = repmat(j1m,1,1,1,3);
j2mr = repmat(j2m,1,1,1,3);
j3mr = repmat(j3m,1,1,1,3);

% nSpinsTest = getRSphere_WholeArray_N(L,Lt,2*NEQ);
% nSpinsTest = permute(nSpinsTest,[1 2 3 5 4]);
% nSpinsTest = bsxfun(@times,occu,nSpinsTest);
% 
% rands = rand(L,L,Lt,2*(NEQ-1),'single','gpuArray');
jrs = zeros(L,L,Lt,3,6,'single','gpuArray');

jrs(:,:,:,:,1) = j1pr;
jrs(:,:,:,:,2) = j2pr;
jrs(:,:,:,:,3) = j3pr;
jrs(:,:,:,:,4) = j1mr;
jrs(:,:,:,:,5) = j2mr;
jrs(:,:,:,:,6) = j3mr;


j = 1;
neqStepSize = 25;

en   = 0;
en2  = 0;
mag  = 0;
mag2 = 0;
mag4 = 0;

for i = 1:NEQ-1
    
    if mod(i,neqStepSize)==1
        nSpinsTest = getRSphere_WholeArray_N(L,Lt,2*NEQ);
        nSpinsTest = permute(nSpinsTest,[1 2 3 5 4]);
        nSpinsTest = bsxfun(@times,occu,nSpinsTest);
        
        rands = samRand(L,L,Lt,2*NEQ);
        j = 1;
    end
    
    
    mxGPU = gather(sum(sBlock(:,:,:,1),'all'));
    myGPU = gather(sum(sBlock(:,:,:,2),'all'));
    mzGPU = gather(sum(sBlock(:,:,:,3),'all'));
    
    %[s1p,~,s2p,~,s3p,~] = updateNeighborSpins(sBlock,L,Lt);
    s1p = sBlock(p1,:,:,:);s1m = sBlock(m1,:,:,:);s2p = sBlock(:,p1,:,:);s2m = sBlock(:,m1,:,:);s3p = sBlock(:,:,tp1,:);s3m = sBlock(:,:,tm1,:);
    %netxyz = bsxfun(@times,j1p,s1p) + bsxfun(@times,j2p,s2p) + bsxfun(@times,j3p,s3p);
    netxyz = j1pr.*s1p + j2pr.*s2p + j3pr.*s3p;
    
    sweepenGPU = sum(netxyz.*(sBlock),'all');
    
    
    sweepen = gather(sweepenGPU);
    
    en_inc  =  sweepen/(L^2*Lt);  %  + sweepen/L3;
    en2_inc = en_inc^2; % + (sweepen/L3)^2;
    
    mag_inc =   sqrt(mxGPU^2 + myGPU^2 + mzGPU^2)/(L^2*Lt);
    
    mag2_inc= mag_inc^2;% +  sweepmag^2;
    mag4_inc = mag_inc^4;% +  sweepmag^4;
    %mag = mag_inc;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  metro sweep 

    nSpins = nSpinsTest(:,:,:,:,j);
    ds = (sBlock - nSpins);
    %s1p = sBlock(p1,:,:,:);s1m = sBlock(m1,:,:,:);s2p = sBlock(:,p1,:,:);s2m = sBlock(:,m1,:,:);s3p = sBlock(:,:,tp1,:);s3m = sBlock(:,:,tm1,:);
%     netxyz = j1pr.*s1p + j2pr.*s2p + j3pr.*s3p + j1mr.*s1m + j2mr.*s2m + j3mr.*s3m;
    netxyz = sum(cat(5,s1p,s2p,s3p,s1m,s2m,s3m).*jrs,5); 
    dE = sum(netxyz.*(ds),4);
    swap = (dE<0 | (exp(-dE*beta) > rands(:,:,:,j)))&checkerboard;
    j=j+1;
    notswap = ~swap;
    sBlock = bsxfun(@times,swap,nSpins) + bsxfun(@times,notswap,sBlock);   
    
    %%%%%%%%%%%%%% Other checkerboard
    nSpins = nSpinsTest(:,:,:,:,j);
    ds = (sBlock - nSpins);
    s1p = sBlock(p1,:,:,:);s1m = sBlock(m1,:,:,:);s2p = sBlock(:,p1,:,:);s2m = sBlock(:,m1,:,:);s3p = sBlock(:,:,tp1,:);s3m = sBlock(:,:,tm1,:);
%     netxyz = j1pr.*s1p + j2pr.*s2p + j3pr.*s3p + j1mr.*s1m + j2mr.*s2m + j3mr.*s3m;
    netxyz = sum(cat(5,s1p,s2p,s3p,s1m,s2m,s3m).*jrs,5); 
    dE = sum(netxyz.*(ds),4);
    swap = (dE<0 | (exp(-dE*beta) > rands(:,:,:,j)))&checkerboardOther;
    j=j+1;
    notswap = ~swap;
    sBlock = bsxfun(@times,swap,nSpins) + bsxfun(@times,notswap,sBlock);
    
    en   = en   + (en_inc   - en)/i;
    en2  = en2  + (en2_inc  - en2)/i;
    mag  = mag  + (mag_inc  - mag)/i;
    mag2 = mag2 + (mag2_inc - mag2)/i;
    mag4 = mag4 + (mag4_inc - mag4)/i;
    
end

%mag  = mag;%  +  sweepmag;
% mag2 = mag^2;% +  sweepmag^2;
% mag4 = mag^4;% +  sweepmag^4;
% mag = mag;
end %subroutine measurement












function jval = jvalue(Flat_Dist,Tail_Heavy_Dist,MIN,MAX,y_exponent)

if Flat_Dist
    if MAX~=MIN
        jval = (samRand*(MAX-MIN) + MIN);
    else
        jval = MAX;
    end
end

if Tail_Heavy_Dist
    jvalue = (samRand^y_exponent);
end


end





function firstPass = getRSphere(i)

startwith = ceil(i/.52) + 150;

firstPass = samRand(startwith,3)-.5;
%thirdPass = rand(startwith,3);
%secondPass = getRSphere2(i);
%gpuPass = rand(startwith,3,'single','gpuArray');



%sizes = vecnorm(firstPass,2,2);
sizes    = sum(firstPass.^2,2);
keepMask = sizes<=.25;
keepsum = cumsum(keepMask);
keepMask = keepMask & keepsum<=i;
firstPass = firstPass(keepMask);
if length(firstPass) < i
    firstPass = getRSphere(i);
else
    sizes = sizes(keepMask);
    sizes = sqrt(sizes);
    sizes = 1./sizes;
    sizes = repmat(sizes,1,3);
    firstPass = firstPass.*sizes;
end


%scatter3(firstPass(:,1),firstPass(:,2),firstPass(:,3))
end




function firstPass = getRSphere_WholeArray(L,Lt)
i = L^2*Lt;

rs = samRand(i,2);
elev = asin(2*rs(:,1)-1);
%elev = asin(rvals);
az = 2*pi*rs(:,2);
%r = 3*(rand(i,1).^(1/3));



%z = sin(elev);
rcoselev = cos(elev);
%x = rcoselev .* cos(az);
%y = rcoselev .* sin(az);
a = rcoselev.*cos(az);
b = rcoselev.*sin(az);
c = sin(elev);
firstPass = [a b c];


firstPass = reshape(firstPass,L,L,Lt,3);
%firstPass = gather(firstPass);
%scatter3(firstPass(:,1),firstPass(:,2),firstPass(:,3))
end




function firstPass = getRSphere_WholeArray_N(L,Lt,N)
i = L^2*Lt*N;

rs = samRand(i,2);
elev = asin(2*rs(:,1)-1);
%elev = asin(rvals);
az = 2*pi*rs(:,2);
%r = 3*(rand(i,1).^(1/3));



%z = sin(elev);
rcoselev = cos(elev);
%x = rcoselev .* cos(az);
%y = rcoselev .* sin(az);
a = rcoselev.*cos(az);
b = rcoselev.*sin(az);
c = sin(elev);
firstPass = [a b c];


firstPass = reshape(firstPass,L,L,Lt,N,3);
%firstPass = gather(firstPass);
%scatter3(firstPass(:,1),firstPass(:,2),firstPass(:,3))
end




function firstPass = getRSphere_WholeArray_Faster(L,Lt)
i = L^2*Lt;

elev = 2.0*(rand(i,1,'single','gpuArray')-.5);
elev = asin(elev);
%elev = asin(rvals);
az = rand(i,1,'single','gpuArray');
az = 2*pi*az;
%r = 3*(rand(i,1).^(1/3));



%z = sin(elev);
rcoselev = cos(elev);
%x = rcoselev .* cos(az);
%y = rcoselev .* sin(az);
a = rcoselev.*cos(az);
rcoselev = rcoselev.*sin(az);
elev = sin(elev);
firstPass = [a rcoselev elev];


firstPass = reshape(firstPass,L,L,Lt,3);
%firstPass = gather(firstPass);
%scatter3(firstPass(:,1),firstPass(:,2),firstPass(:,3))
end


function firstPass = getRSphere_notGPU(L,Lt)
i = L^2*Lt;

rs = rand(i,2,'single');
elev = asin(2*rs(:,1)-1);
%elev = asin(rvals);
az = 2*pi*rs(:,2);
%r = 3*(rand(i,1).^(1/3));



%z = sin(elev);
rcoselev = cos(elev);
%x = rcoselev .* cos(az);
%y = rcoselev .* sin(az);
a = rcoselev.*cos(az);
b = rcoselev.*sin(az);
c = sin(elev);
firstPass = [a b c];


%firstPass = reshape(firstPass,3,1);
%firstPass = gather(firstPass);
%scatter3(firstPass(:,1),firstPass(:,2),firstPass(:,3))
end


function [m1,m2,m3,m4,m5,m6] = getNeighborTable(L,Lt,L2,L3)
m1 = zeros(L3,1);            % neighbor table
m2 = zeros(L3,1);
m3 = zeros(L3,1);
m4 = zeros(L3,1);
m5 = zeros(L3,1);
m6 = zeros(L3,1);


for i1=1:Lt
    for i2=1:L
        for i3=1:L
            
            is = L2*(i1-1) + L*(i2-1) + i3;
            
            if (i1==Lt)
                m1(is)=is - L2*(Lt-1);
            else
                m1(is)=is + L2;
            end
            
            if (i1==1)
                m2(is)=is + L2*(Lt-1);
            else
                m2(is)=is - L2;
            end
            
            if (i2==L)
                m3(is)= is - L*(L-1);
            else
                m3(is)= is + L;
            end
            
            if (i2==1)
                m4(is)=is + L*(L-1);
            else
                m4(is)=is - L;
            end
            
            if (i3==L)
                m5(is)= is - (L-1);
            else
                m5(is)= is + 1;
            end
            
            if (i3==1)
                m6(is)= is + (L-1);
            else
                m6(is)= is - 1;
            end
        end
    end
end



end

function [j1,j3,j5] = initializeSiteCouplings(L,Lt,L2,L3,Flat_Dist,Tail_Heavy_Dist,MIN,MAX,y_exponent)
j1 = zeros(L3,1);
j3 = zeros(L3,1);
j5 = zeros(L3,1);


for ij3=1:L % note site naming convention breaks from rest of program for this loop
    for ij1=1:L
        ij = ij1+L*(ij3-1);
        j1(ij)=1;
        j3(ij)=jvalue(Flat_Dist,Tail_Heavy_Dist,MIN,MAX,y_exponent);
        j5(ij)=jvalue(Flat_Dist,Tail_Heavy_Dist,MIN,MAX,y_exponent);
        
        ij5=2:Lt;
        thisS = ij+(ij5-1)*L2;
        j1(thisS)=j1(ij);
        j3(thisS)=j3(ij);
        j5(thisS)=j5(ij);
    end
end

end



function [j1p,j1m,j2p,j2m,j3p,j3m] = initializeSiteCouplings_Block(L,Lt,Flat_Dist,Tail_Heavy_Dist,MIN,MAX,y_exponent)


j1p = zeros(L,L,Lt,'single');
%j1m = zeros(L,L,Lt,'single');
j2p = zeros(L,L,Lt,'single');
%j2m = zeros(L,L,Lt,'single');
j3p = zeros(L,L,Lt,'single');
%j3m = zeros(L,L,Lt,'single');



for i=1:L
    for j=1:L
        j1p(i,j,:) = jvalue(Flat_Dist,Tail_Heavy_Dist,MIN,MAX,y_exponent);
        j2p(i,j,:) = jvalue(Flat_Dist,Tail_Heavy_Dist,MIN,MAX,y_exponent);
        j3p(i,j,:) = 1;%jvalue(Flat_Dist,Tail_Heavy_Dist,MIN,MAX,y_exponent);
    end
end

j1m = circshift(j1p,1,1);
j2m = circshift(j2p,1,2);
j3m = circshift(j3p,1,3);

end

function occu = initializeSiteImpurities(L,Lt,impconc)
occu = zeros(L,L,Lt,'logical');
rs = samRand(L, L);

rs = rs > impconc;
for i = 1:L
    for j = 1:L
        occu(i,j,:) = rs(i,j);
    end
end

end


function checkerboard = generateCheckerboard(L,Lt)


checkerboard = zeros(L,L,Lt,'logical');

for i = 1:L
    for j = 1:L
        for k = 1:Lt
            checkerboard(i,j,k) = mod(i+j+k,2) == 0;
        end
    end
end

end

function [s1p,s1m,s2p,s2m,s3p,s3m] = updateNeighborSpins(s,L,Lt)

sizes = [L L Lt 3];
s1p = circshift_Op(s,-1,1,sizes);
s1m = circshift_Op(s,1,1,sizes);
s2p = circshift_Op(s,-1,2,sizes);
s2m = circshift_Op(s,1,2,sizes);
s3p = circshift_Op(s,-1,3,sizes);
s3m = circshift_Op(s,1,3,sizes);

% s1p = circshift(s,-1,1);
% s1m = circshift(s,1,1);
% s2p = circshift(s,-1,2);
% s2m = circshift(s,1,2);
% s3p = circshift(s,-1,3);
% s3m = circshift(s,1,3);
end







function b = circshift_Op(a,p,dim,s)


numDimsA =4;
p = [zeros(1,dim-1,'like',p) p];

% Make sure the shift vector has the same length as numDimsA.
% The missing shift values are assumed to be 0. The extra
% shift values are ignored when the shift vector is longer
% than numDimsA.
if (numel(p) < numDimsA)
    p(numDimsA) = 0;
end

% Calculate the indices that will convert the input matrix to the desired output
% Initialize the cell array of indices
idx = cell(1, numDimsA);
% Loop through each dimension of the input matrix to calculate shifted indices

for k = 1:numDimsA
    if p(k) == 0
        idx{k} = ':';
    else
%         m      = size(a,k);
        m = s(k);
        
        idx{k} = mod((0:m-1)-double(rem(p(k),m)), m)+1;
    end
end




% Perform the actual conversion by indexing into the input matrix

b = a(idx{1},idx{2},idx{3},idx{4});
% b = a(idx{:});
end


%[sBlock] = metro_sweep_faster(sBlock,L,Lt,beta,occu,j1p,j1m,j2p,j2m,j3p,j3m,checkerboard)
function [sBlock] = wolff_sweep(NCLUSTER,sBlock,L,Lt,beta,occu,j1p,j1m,j2p,j2m,j3p,j3m,checkerboard)
%    Performs a Wolff sweep consisting of several single cluster flips

% integer(i4b)    :: stack(0:L3-1)         % stack for cluster construction
% integer(i4b)    :: sp                    % stackpointer
% logical(ilog)   :: oldspin,newspin       % temp. spin variables
% integer(i4b)    :: current               % current site in cluster construction
% integer(i4b)    :: isize                 % size of current cluster
% integer(i4b)    :: nflip                 % number of flipped spins
% integer(i4b)    :: icluster              % cluster index
% real(r8b)       :: nx,ny,nz              % reflection direction
% real(r8b)       :: scalar1,scalar2, snsn          % scalar products in addition probability
% real(r8b)       :: padd,help

wrapN = @(x) [(1 + mod(x(1)-1, L)) (1 + mod(x(2)-1, L)) (1 + mod(x(3)-1, Lt))];

stack = zeros(L^2*Lt,3,'gpuArray');


nflip = 0;
icluster = 0;

while icluster < NCLUSTER
    %is = int(L3*rlfsr113());             % seed site for Wolff cluster
    
    is = [randi(L,1,2) randi(Lt,1,1)];  % seed site for Wolff cluster
    
    if occu(is(1),is(2),is(3))                   % is site occupied?
        %[s1p,s1m,s2p,s2m,s3p,s3m] = updateNeighborSpins(sBlock,L,Lt);
        [o1p,o1m,o2p,o2m,o3p,o3m] = updateNeighborSpins(occu);
        icluster = icluster + 1;
        
        sp = 1;
        stack(sp,:) = is;
        
        
        oldSpins = squeeze(sBlock(is(1),is(2),is(3),:));
        nSpins = squeeze(getRSphere_WholeArray(1,1));
        
        
        isize = 1;
        %scalar2 = (nx*sx(is)+ny*sy(is)+nz*sz(is));               % scalar product for p_add
        scalar2 = dot(nSpins,oldSpins);               % scalar product for p_add
        
        %         sx(is) = sx(is)-2*nx*scalar2;                              % now flip seed spin
        %         sy(is) = sy(is)-2*ny*scalar2;
        %         sz(is) = sz(is)-2*nz*scalar2;
        
        sBlock(is(1),is(2),is(3),:) = oldSpins - 2*scalar2*nSpins;
        
        
        while sp > 0                   % now build the cluster
            c = stack(sp,:);              % get site from stack
            cS = squeeze(sBlock(c(1),c(2),c(3),:));
            %scalar1 = -(nx*sx(current)+ny*sy(current)+nz*sz(current));                % scalar product for p_add
            scalar1 = -dot(nSpins,cS);
            
            sp = sp - 1;
            
            if o1p(c(1),c(2),c(3))
                %lo = wrapN([ c(1)+1 c(2) c(3)]);
                lo = wrapM(c(1)+1,c(2),c(3),L,Lt);
                cS = [sBlock(lo(1),lo(2),lo(3),1); sBlock(lo(1),lo(2),lo(3),2); sBlock(lo(1),lo(2),lo(3),3)];
                scalar2 =  dot(nSpins,cS);   % scalar product for p_add
                snsn = scalar1*scalar2;
                if snsn>0
                    padd = 1.0-exp(-2*j1p(c(1),c(2),c(3))*beta*snsn);                                                 % check whether parallel
                    help = rand(1,1,'single','gpuArray');
                    if help < padd
                        
                        %%% NEED TO ACTUALLY UPDATE THE SPIN
                        sBlock(lo(1),lo(2),lo(3),:) = cS - 2*scalar2*nSpins;
                        
                        sp = sp+1;
                        stack(sp,:) = lo;
                        
                        isize = isize+1;
                    end
                end
            end
            
            if o1m(c(1),c(2),c(3))
                lo = wrapN([ c(1)-1 c(2) c(3)]);
                cS = [sBlock(lo(1),lo(2),lo(3),1); sBlock(lo(1),lo(2),lo(3),2); sBlock(lo(1),lo(2),lo(3),3)];
                scalar2 =  dot(nSpins,cS);   % scalar product for p_add
                snsn = scalar1*scalar2;
                if snsn>0
                    padd = 1.0-exp(-2*j1m(c(1),c(2),c(3))*beta*snsn);                                                 % check whether parallel
                    help = rand(1,1,'single','gpuArray');
                    if help < padd
                        
                        %%% NEED TO ACTUALLY UPDATE THE SPIN
                        sBlock(lo(1),lo(2),lo(3),:) = cS - 2*scalar2*nSpins;
                        
                        sp = sp+1;
                        stack(sp,:) = lo;
                        
                        isize = isize+1;
                    end
                end
            end
            
            if o2p(c(1),c(2),c(3))
                lo = wrapN([ c(1) c(2)+1 c(3)]);
                cS = [sBlock(lo(1),lo(2),lo(3),1); sBlock(lo(1),lo(2),lo(3),2); sBlock(lo(1),lo(2),lo(3),3)];
                scalar2 =  dot(nSpins,cS);   % scalar product for p_add
                snsn = scalar1*scalar2;
                if snsn>0
                    padd = 1.0-exp(-2*j2p(c(1),c(2),c(3))*beta*snsn);                                                 % check whether parallel
                    help = rand(1,1,'single','gpuArray');
                    if help < padd
                        
                        %%% NEED TO ACTUALLY UPDATE THE SPIN
                        sBlock(lo(1),lo(2),lo(3),:) = cS - 2*scalar2*nSpins;
                        
                        sp = sp+1;
                        stack(sp,:) = lo;
                        
                        isize = isize+1;
                    end
                end
            end
            
            if o2m(c(1),c(2),c(3))
                lo = wrapN([ c(1) c(2)-1 c(3)]);
                cS = [sBlock(lo(1),lo(2),lo(3),1); sBlock(lo(1),lo(2),lo(3),2); sBlock(lo(1),lo(2),lo(3),3)];
                scalar2 =  dot(nSpins,cS);   % scalar product for p_add
                snsn = scalar1*scalar2;
                if snsn>0
                    padd = 1.0-exp(-2*j2m(c(1),c(2),c(3))*beta*snsn);                                                 % check whether parallel
                    help = rand(1,1,'single','gpuArray');
                    if help < padd
                        
                        %%% NEED TO ACTUALLY UPDATE THE SPIN
                        sBlock(lo(1),lo(2),lo(3),:) = cS - 2*scalar2*nSpins;
                        
                        sp = sp+1;
                        stack(sp,:) = lo;
                        
                        isize = isize+1;
                    end
                end
            end
            
            if o3p(c(1),c(2),c(3))
                lo = wrapN([ c(1) c(2) c(3)+1]);
                cS = [sBlock(lo(1),lo(2),lo(3),1); sBlock(lo(1),lo(2),lo(3),2); sBlock(lo(1),lo(2),lo(3),3)];
                scalar2 =  dot(nSpins,cS);   % scalar product for p_add
                snsn = scalar1*scalar2;
                if snsn>0
                    padd = 1.0-exp(-2*j3p(c(1),c(2),c(3))*beta*snsn);                                                 % check whether parallel
                    help = rand(1,1,'single','gpuArray');
                    if help < padd
                        
                        %%% NEED TO ACTUALLY UPDATE THE SPIN
                        sBlock(lo(1),lo(2),lo(3),:) = cS - 2*scalar2*nSpins;
                        
                        sp = sp+1;
                        stack(sp,:) = lo;
                        
                        isize = isize+1;
                    end
                end
            end
            
            if o3m(c(1),c(2),c(3))
                lo = wrapN([ c(1) c(2) c(3)-1]);
                cS = [sBlock(lo(1),lo(2),lo(3),1); sBlock(lo(1),lo(2),lo(3),2); sBlock(lo(1),lo(2),lo(3),3)];
                scalar2 =  dot(nSpins,cS);   % scalar product for p_add
                snsn = scalar1*scalar2;
                if snsn>0
                    padd = 1.0-exp(-2*j3m(c(1),c(2),c(3))*beta*snsn);                                                 % check whether parallel
                    help = rand(1,1,'single','gpuArray');
                    if help < padd
                        
                        %%% NEED TO ACTUALLY UPDATE THE SPIN
                        sBlock(lo(1),lo(2),lo(3),:) = cS - 2*scalar2*nSpins;
                        
                        sp = sp+1;
                        stack(sp,:) = lo;
                        
                        isize = isize+1;
                    end
                end
            end
            
            
        end       % of cluster building and flipping
        
        nflip   = nflip + isize;
        
    end          % of if(occu(is))
end          % of do icluster
end

function x = wrapM(x,L,Lt)
x1 = x(1);
x2 = x(2);
x3 = x(3);
if x1 == 0 || x2 == 0 || x3 == 0 || x1 > L || x2 > L  || x3 >Lt
    x = [(1 + mod(x1-1, L)) (1 + mod(x2-1, L)) (1 + mod(x3-1, Lt))];
else
    
end
end


%[sBlock] = metro_sweep_faster(sBlock,L,Lt,beta,occu,j1p,j1m,j2p,j2m,j3p,j3m,checkerboard)
function [sBlock] = wolff_sweep_notGPU(NCLUSTER,sBlock,L,Lt,beta,occu,j1p,j1m,j2p,j2m,j3p,j3m)
%    Performs a Wolff sweep consisting of several single cluster flips

% [o1p,o1m,o2p,o2m,o3p,o3m] = updateNeighborSpins(occu);
[o1p,o1m,o2p,o2m,o3p,o3m] = updateNeighborSpins(occu,L,Lt);
wrapN = @(x) [(1 + mod(x(1)-1, L)) (1 + mod(x(2)-1, L)) (1 + mod(x(3)-1, Lt))];
wrapNv = @(x) [(1 + mod(x(:,1)-1, L)) (1 + mod(x(:,2)-1, L)) (1 + mod(x(:,3)-1, Lt))];

wrapNv1 = @(x) (1 + mod(x(:,1)-1, L)) ;


c1p = (1:L)' + 1;
c1p(end) = 1;
c1m= (1:L)' - 1;
c1m(1) = L;

c2p = c1p;
c2m = c1m;

c3p = (1:Lt)' + 1;
c3p(end) = 1;
c3m= (1:Lt)' - 1;
c3m(1) = Lt;







stack = zeros(L^2*Lt,3);

%sBlock = gather(sBlock);
occu = gather(occu);
j1p  = gather(j1p);
j1m  = gather(j1m);
j2p  = gather(j2p);
j2m  = gather(j2m);
j3p  = gather(j3p);
j3m  = gather(j3m);


nflip = 0;
icluster = 0;

while icluster < NCLUSTER
    %is = int(L3*rlfsr113());             % seed site for Wolff cluster
    addedTo = zeros(L,L,Lt,'logical');
    is = [randi(L,1,2) randi(Lt,1,1)];  % seed site for Wolff cluster
    
    if occu(is(1),is(2),is(3))                   % is site occupied?
        %[s1p,s1m,s2p,s2m,s3p,s3m] = updateNeighborSpins(sBlock,L,Lt);
        addedTo(is(1),is(2),is(3)) = 1;
        icluster = icluster + 1;
        
        sp = 1;
        stack(sp,:) = is;
        
        
        %oldSpins = squeeze(sBlock(is(1),is(2),is(3),:));
        %oldSpins = reshape(sBlock(is(1),is(2),is(3),:),3,1);
        
        %aa = shiftdim(sBlock(is(1),is(2),is(3),:),3);
        oldSpins = [sBlock(is(1),is(2),is(3),1); sBlock(is(1),is(2),is(3),2); sBlock(is(1),is(2),is(3),3)];
        nSpins = getRSphere_notGPU(1,1)';
        
        
        isize = 1;
        %scalar2 = (nx*sx(is)+ny*sy(is)+nz*sz(is));               % scalar product for p_add
        scalar2 = dot(nSpins,oldSpins);               % scalar product for p_add
        
        %         sx(is) = sx(is)-2*nx*scalar2;                              % now flip seed spin
        %         sy(is) = sy(is)-2*ny*scalar2;
        %         sz(is) = sz(is)-2*nz*scalar2;
        
        sBlock(is(1),is(2),is(3),:) = oldSpins - 2*scalar2*nSpins;
        helpS = rand(6,1,'single');
        
        while sp > 0                   % now build the cluster
            c = stack(sp,:);              % get site from stack
            %cS = squeeze(sBlock(c(1),c(2),c(3),:));
            cS = [sBlock(c(1),c(2),c(3),1); sBlock(c(1),c(2),c(3),2); sBlock(c(1),c(2),c(3),3)];
            %scalar1 = -(nx*sx(current)+ny*sy(current)+nz*sz(current));                % scalar product for p_add
            scalar1 = -dot(nSpins,cS);
            
            sp = sp - 1;
            c1 = c(1); % Current site locations
            c2 = c(2);
            c3 = c(3);
            allC = [c1p(c1) c2 c3;c1m(c1) c2 c3;c1 c2p(c2) c3;c1 c2m(c2) c3;c1 c2 c3p(c3);c1 c2 c3m(c3)];
            %allC = wrapNv(allC);
            if o1p(c(1),c(2),c(3)) && ~(addedTo(allC(1,1),allC(1,2),allC(1,3)))
                %lo = wrapN([ c(1)+1 c(2) c(3)]);
                lo = allC(1,:);
                %cS = squeeze(sBlock(lo(1),lo(2),lo(3),:));
                cS = [sBlock(lo(1),lo(2),lo(3),1); sBlock(lo(1),lo(2),lo(3),2); sBlock(lo(1),lo(2),lo(3),3)];
                scalar2 =  dot(nSpins,cS);   % scalar product for p_add
                snsn = scalar1*scalar2;
                if snsn>0
                    padd = 1.0-exp(-2*j1p(c(1),c(2),c(3))*beta*snsn);                                                 % check whether parallel
                    help = helpS(1);
                    if help < padd
                        
                        %%% NEED TO ACTUALLY UPDATE THE SPIN
                        sBlock(lo(1),lo(2),lo(3),:) = cS - 2*scalar2*nSpins;
                        addedTo(lo(1),lo(2),lo(3)) = 1;
                        sp = sp+1;
                        stack(sp,:) = lo;
                        
                        isize = isize+1;
                    end
                end
            end
            
            if o1m(c(1),c(2),c(3))&& ~(addedTo(allC(2,1),allC(2,2),allC(2,3)))
                lo = allC(2,:);
                cS = [sBlock(lo(1),lo(2),lo(3),1); sBlock(lo(1),lo(2),lo(3),2); sBlock(lo(1),lo(2),lo(3),3)];
                scalar2 =  dot(nSpins,cS);   % scalar product for p_add
                snsn = scalar1*scalar2;
                if snsn>0
                    padd = 1.0-exp(-2*j1m(c(1),c(2),c(3))*beta*snsn);                                                 % check whether parallel
                    help = helpS(2);
                    if help < padd
                        
                        %%% NEED TO ACTUALLY UPDATE THE SPIN
                        sBlock(lo(1),lo(2),lo(3),:) = cS - 2*scalar2*nSpins;
                        addedTo(lo(1),lo(2),lo(3)) = 1;
                        sp = sp+1;
                        stack(sp,:) = lo;
                        
                        isize = isize+1;
                    end
                end
            end
            
            if o2p(c(1),c(2),c(3))&& ~(addedTo(allC(3,1),allC(3,2),allC(3,3)))
                lo = allC(3,:);
                cS = [sBlock(lo(1),lo(2),lo(3),1); sBlock(lo(1),lo(2),lo(3),2); sBlock(lo(1),lo(2),lo(3),3)];
                scalar2 =  dot(nSpins,cS);   % scalar product for p_add
                snsn = scalar1*scalar2;
                if snsn>0
                    padd = 1.0-exp(-2*j2p(c(1),c(2),c(3))*beta*snsn);                                                 % check whether parallel
                    help = helpS(3);
                    if help < padd
                        
                        %%% NEED TO ACTUALLY UPDATE THE SPIN
                        sBlock(lo(1),lo(2),lo(3),:) = cS - 2*scalar2*nSpins;
                        addedTo(lo(1),lo(2),lo(3)) = 1;
                        sp = sp+1;
                        stack(sp,:) = lo;
                        
                        isize = isize+1;
                    end
                end
            end
            
            if o2m(c(1),c(2),c(3))&& ~(addedTo(allC(4,1),allC(4,2),allC(4,3)))
                lo = allC(4,:);
                cS = [sBlock(lo(1),lo(2),lo(3),1); sBlock(lo(1),lo(2),lo(3),2); sBlock(lo(1),lo(2),lo(3),3)];
                scalar2 =  dot(nSpins,cS);   % scalar product for p_add
                snsn = scalar1*scalar2;
                if snsn>0
                    padd = 1.0-exp(-2*j2m(c(1),c(2),c(3))*beta*snsn);                                                 % check whether parallel
                    help = helpS(4);
                    if help < padd
                        
                        %%% NEED TO ACTUALLY UPDATE THE SPIN
                        sBlock(lo(1),lo(2),lo(3),:) = cS - 2*scalar2*nSpins;
                        addedTo(lo(1),lo(2),lo(3)) = 1;
                        sp = sp+1;
                        stack(sp,:) = lo;
                        
                        isize = isize+1;
                    end
                end
            end
            
            if o3p(c(1),c(2),c(3))&& ~(addedTo(allC(5,1),allC(5,2),allC(5,3)))
                lo = allC(5,:);
                cS = [sBlock(lo(1),lo(2),lo(3),1); sBlock(lo(1),lo(2),lo(3),2); sBlock(lo(1),lo(2),lo(3),3)];
                scalar2 =  dot(nSpins,cS);   % scalar product for p_add
                snsn = scalar1*scalar2;
                if snsn>0
                    padd = 1.0-exp(-2*j3p(c(1),c(2),c(3))*beta*snsn);                                                 % check whether parallel
                    help = helpS(5);
                    if help < padd
                        
                        %%% NEED TO ACTUALLY UPDATE THE SPIN
                        sBlock(lo(1),lo(2),lo(3),:) = cS - 2*scalar2*nSpins;
                        addedTo(lo(1),lo(2),lo(3)) = 1;
                        sp = sp+1;
                        stack(sp,:) = lo;
                        
                        isize = isize+1;
                    end
                end
            end
            
            if o3m(c(1),c(2),c(3))&& ~(addedTo(allC(6,1),allC(6,2),allC(6,3)))
                lo = allC(6,:);
                cS = [sBlock(lo(1),lo(2),lo(3),1); sBlock(lo(1),lo(2),lo(3),2); sBlock(lo(1),lo(2),lo(3),3)];
                scalar2 =  dot(nSpins,cS);   % scalar product for p_add
                snsn = scalar1*scalar2;
                if snsn>0
                    padd = 1.0-exp(-2*j3m(c(1),c(2),c(3))*beta*snsn);                                                 % check whether parallel
                    help = helpS(6);
                    if help < padd
                        %%% NEED TO ACTUALLY UPDATE THE SPIN
                        sBlock(lo(1),lo(2),lo(3),:) = cS - 2*scalar2*nSpins;
                        addedTo(lo(1),lo(2),lo(3)) = 1;
                        sp = sp+1;
                        stack(sp,:) = lo;
                        
                        isize = isize+1;
                    end
                end
            end
            
            
        end       % of cluster building and flipping
        
        nflip   = nflip + isize;
        
    end          % of if(occu(is))
end          % of do icluster
fprintf('Temperature = %.4f, cluster size = %.3f\n',1/beta,nflip/(NCLUSTER*    L^2*Lt))
sBlock = gpuArray(sBlock);

end










