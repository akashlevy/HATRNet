function [f,Y] = fft_freq(y,Fs,n,indexStart,indexEnd,highPassFlag)

    % If length of DFT is not specified, assume the default (length of time
    % domain vector, y)
    if nargin <= 2, n = length(y); end
    if nargin < 6, highPassFlag = 0; end
    
    % High pass frequency at 10 Hz if desired by user
    if highPassFlag, y = apply_highPassFilter_butter(y,Fs,10); end;

    xdft = fft(y,n); % Perform Fast Fourier Transform (FFT)
    xdft = xdft(1:floor(length(xdft)/2+1));
    xdft = 1/length(y).*xdft;
    xdft(2:end-1) = 2*xdft(2:end-1);
    Y = abs(xdft);
    f = [0:Fs/n:Fs/2]';
    
    % Specify a starting and ending index (when you want to specify a range
    % of frequencies to plot, rather than all that are comptued by the
    % function
    if nargin > 3
        Y = Y(indexStart:indexEnd);
        f = f(indexStart:indexEnd);
    end
        
end
