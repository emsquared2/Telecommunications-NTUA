% Emmanouilidis Emmanouil 
% AM: 03119435  -> 4 + 3 + 5 = 12 -> 1 + 2 = 3
%----------------------------
% Litsos Ioannis 
% AM: 03119135  -> 1 + 3 + 5 = 9 
%----------------------------

%Group: 80 

%Code by Emmanouilidis Emmanouil

% ######################## EXERCISE 1 ########################

A = 4;
fm = 3000;
%fm = 9000;

T = 4 * (1/fm);

fs0 = 1000000;
t0 = 0:1/fs0:T-1/fs0;

x0 = A*sawtooth(2*pi*fm*t0, 1/2);

figure(1)
plot(t0,x0)
title('Triangular Pulse')
xlabel ('Time [s]')
ylabel('Amplitude [V]')
grid on

% 1a)

fs1 = 30*fm;
fs2 = 50*fm;
dt1 = 1/fs1;
dt2 = 1/fs2;
t1 = 0:dt1:T-1/fs0;
t2 = 0:dt2:T-1/fs0;
x1 = A*sawtooth(2*pi*fm*t1, 1/2);
x2 = A*sawtooth(2*pi*fm*t2, 1/2);

% a (i)

figure(2);
stem(t1,x1,'rx-')
title("Sampled Pulse (fs1 = 30 * fm)")
xlabel ('Time [s]')
ylabel('Amplitude [V]')
legend('fs1 = 30 * fm')
grid on

% a (ii)

figure(3)
stem(t2,x2, 'bo-')
title("Sampled Pulse (fs2 = 50 * fm)")
xlabel ('Time [s]')
ylabel('Amplitude [V]')
legend('fs2 = 50 * fm')
grid on

% a (iii)

figure(4)
hold on
stem(t1,x1,'rx-')
stem(t2,x2,'bo-')
title("Both Sampled Pulses")
xlabel ('Time [s]')
ylabel('Amplitude [V]')
legend('fs1 = 30 * fm' , 'fs2 = 50 * fm')
grid on
hold off

% 1b)

fs=4*fm;
dts = 1/fs;
t_s = 0:dts:T-1/fs0;
x_s = A*sawtooth(2*pi*fm*t_s, 1/2);

figure(5)
stem(t_s,x_s,'rx-')
title("Sampled Pulse (fs = 4 * fm)")
xlabel ('Time [s]')
ylabel('Amplitude [V]')
legend('fs = 4 * fm')
grid on

% DFT of the triangular pulse
Y = fft(x0);
N=length(Y);
f_y = 0:fs0/N:fs0/2-fs0/N;
figure (6)
plot(f_y, abs(Y(1:(N-1)/2))); % DFT is symmetrical, we plot only half of the transform
xlabel('Frequency (Hz)');
ylabel('Amplitude');
title('Phase spectrum of the triangular pulse');
xlim([0 40000])

% 1c)

A_c = 1;
z = A_c*sin(2*pi*fm*t0);
figure(7)
plot(t0,z)
title('z(t) = sin(2πfmt)')
xlabel ('Time [s]')
ylabel('Amplitude [V]')
grid on

% c (i)

% 1c (i)->a

z1 = A_c*sin(2*pi*fm*t1);

figure(8);
stem(t1,z1,'rx-')
title("Sampled Signal z(t) (fs1 = 30 * fm)")
xlabel ('Time [s]')
ylabel('Amplitude [V]')
legend('fs1 = 30 * fm')
grid on


z2 = A_c*sin(2*pi*fm*t2);

figure(9)
stem(t2,z2, 'bo-')
title("Sampled Signal z(t) (fs2 = 50 * fm)")
xlabel ('Time [s]')
ylabel('Amplitude [V]')
legend('fs2 = 50 * fm')
grid on


figure(10)
hold on
stem(t1,z1,'rx-')
stem(t2,z2,'bo-')
title("z(t) sampled with fs1 & fs2")
xlabel ('Time [s]')
ylabel('Amplitude [V]')
legend('fs1 = 30 * fm' , 'fs2 = 50 * fm')
grid on
hold off

% 1c (i) -> b

z_s = A_c*sin(2*pi*fm*t_s);

figure(11)
stem(t_s,z_s,'rx-')
title("Sampled Signal z(t) (fs = 4 * fm)")
xlabel ('Time [s]')
ylabel('Amplitude [V]')
legend('fs = 4 * fm')
grid on

% c (ii)

fm_gcd = gcd(fm, fm+1000);  % new frequency is equal to
                            % gcd(fm, fm+1000)
T_gcd = 1/fm_gcd;   
t_gcd = 0:1/fs0:T_gcd-1/fs0;
q = sin(2*pi*fm*t_gcd) + sin(2*pi*(fm+1000)*t_gcd);
figure(12)
plot(t_gcd, q)
grid on
title('q(t) = z(t) + sin(2π(fm+1000)t)')
xlabel('Time [s]')
ylabel('Amplitude [V]')

% 1c (ii)-> a

t1_c = 0:dt1:T_gcd-1/fs0;
q1 = sin(2*pi*fm*t1_c) + sin(2*pi*(fm + 1000)*t1_c);
figure (13)
stem(t1_c, q1, 'rx-')
grid on 
title('Sampled q(t) (fs1 = 30 * fm)')
xlabel('Time [s]')
ylabel('Amplitude [V]')

t2_c = 0:dt2:T_gcd-1/fs0;
q2 = sin(2*pi*fm*t2_c) + sin(2*pi*(fm + 1000)*t2_c);
figure (14)
stem(t2_c, q2, 'bo-')
grid on 
title('Sampled q(t) (fs2 = 50 * fm)')
xlabel('Time [s]')
ylabel('Amplitude [V]')

figure (15)
hold on
stem(t1_c, q1, 'rx-')
stem(t2_c, q2, 'bo-')
title('q(t) sampled with fs1 & fs2')
xlabel('Time [s]')
ylabel('Amplitude [V]')
legend('fs1 = 30 * fm','fs2 = 50 * fm');
grid on
hold off

% 1c (ii)-> b

t_c_ii_b = 0:dts:T_gcd-1/fs0;
q_b = sin(2*pi*fm*t_c_ii_b) + sin(2*pi*(fm + 1000)*t_c_ii_b);
figure (16)
stem(t_c_ii_b, q_b, 'bo-')
grid on 
title('Sampled q(t) (fs = 4 * fm)')
xlabel('Time [s]')
ylabel('Amplitude [V]')


%-------------------------------------------------------------------

% ######################## EXERCISE 2 ########################

% 2.a)

x_triangle_1 = x1;
Max_Value_Dec = 32;  % 2^5 = 32
dec_sequence = (0:Max_Value_Dec-1);
gray_sequence = qammod(dec_sequence, Max_Value_Dec, 'gray');

[gray_sec,map_gray] = bin2gray(dec_sequence,'qam',Max_Value_Dec);

gray_code = dec2bin(gray_sec);

R=5;      %bits
L = 2^R ; %number of quantized levels
D = (max(x_triangle_1) - min(x_triangle_1))/L; %step size
I = round((x_triangle_1 - min(x_triangle_1))/D) ; %index
x_quantized = min(x_triangle_1) + I*D;
figure(17);
stem (t1,x_quantized,'rx-');
grid on
title('Quantized triangle with fs1 = 30fm')
xlabel('Time [s]')
ylabel('Gray Code')
yticks([-4 -3.75 -3.5 -3.25 -3 -2.75 -2.5 -2.25 -2 -1.75 -1.5 -1.25 -1 -0.75 -0.5 -0.25 0.25 0.5 0.75 1 1.25 1.5 1.75 2 2.25 2.5 2.75 3 3.25 3.5 3.75 4])
yticklabels(gray_code);

% 2.b)

% 2.b (i)

q = x_triangle_1 - x_quantized;   % error = (sampled signal) - (quantized signal) 
error_first10 = q(1,1:10);          
std_first10 = std(error_first10); 

% 2.b (ii)

error_first20 = q(1,1:20);
std_first20 = std(error_first20);


% 2.b (iii)

% We calculate SNR for the first 10 & 20 samples and 
% theoretical SNR by using the following formula:
% SNRq(in dB) = 10log10((3*2^(2*R))/ F^2) where
% F = max(x_triangle_1)/rms(x_triangle_1)

numerator = 3*2^(2*R);

% SNR for first 10 samples
rms_first10 = rms(x_triangle_1(1:10));
F_first10 = max(x_triangle_1(1:10))/rms_first10;
SNR10 = 10*log10(numerator./(F_first10^2));

% SNR for first 20 samples
rms_first20 = rms(x_triangle_1(1:20));
F_first20 = max(x_triangle_1(1:20))/rms_first20;
SNR20 = 10*log10(numerator./(F_first20^2));

% Theoretical SNR
rms_x_triangle_1 = rms(x_triangle_1);
F_x_triangle_1 = max(x_triangle_1)/rms_x_triangle_1;
SNR_Theoretical = 10*log10(numerator./(F_x_triangle_1^2));


% 2.c 

bit_stream = char(zeros(30,5));
for i = 1:30
    if x_quantized(i) < 0
        index_gc = ( x_quantized(i) + 4 ) / 0.25 + 1;
        bit_stream(i,:) = gray_code(index_gc,:);
    else 
        index_gc = ( x_quantized(i) + 4 ) / 0.25;
        bit_stream(i,:) = gray_code(index_gc,:);
    end
end


x_axis_bit_stream = []; 
y_axis_bit_stream = []; 
index_bs = 1;
index_xy = 1;
for i = 1:30
    for j = 1:5
        x_axis_bit_stream(index_xy:index_xy+3) =  [2*(index_bs-1) 2*(index_bs-1+0.5) 2*(index_bs-1+0.5) 2*(index_bs)];
        if(bit_stream(i,j) == '0')
            y_axis_bit_stream(index_xy:index_xy+3) = [-fm/1000 -fm/1000 0 0];
            index_xy = index_xy + 4;
        else 
            y_axis_bit_stream(index_xy:index_xy+3) = [ fm/1000 fm/1000 0 0];
            index_xy = index_xy + 4;
        end
        index_bs = index_bs + 1;
    end
end

figure (18)
plot(x_axis_bit_stream, y_axis_bit_stream), axis([0, 300, (-fm/1000)-1, (fm/1000)+1]);
title('Polar RZ')
xlabel('Time [ms]')
ylabel('Ampltidue [V]')

%------------------------------------------------------------------------------------

% ######################## EXERCISE 3 ########################


% random 36 bit string
random_bit_string = randi([0,1],[36,1]);


bit_string_len = length(random_bit_string); % Length of Bit String
Tb = 0.25;   % Bit period
fc = 2;   % Carrier frequency
nb = 100; % Digital Signal per Bit

% Plotting the Random Digital String
random_bit_digital_string = [];
for i = 1:bit_string_len
    if random_bit_string(i) == 1
        bit_signal = ones(1, nb);
    else 
        bit_signal = zeros(1, nb);
    end
    random_bit_digital_string = [random_bit_digital_string bit_signal];
end

t_bitstream = Tb/nb:Tb/nb:bit_string_len*Tb;
figure (19)
grid on
plot(t_bitstream, random_bit_digital_string, 'LineWidth', 2.5);
xlabel('Time[s]')
ylim([-0.5 1.5])
ylabel('Amplitude[V]')
title('Digital Signal of Random Bit String') 

% 3.a

% BPSK Modulation
t_1bitperiod = Tb/nb:Tb/nb:Tb; % The length of 1 bit is 0.25sec (Tb)
mod_BPSK = [];
for i = 1:bit_string_len
    if random_bit_string(i) == 1
        bit_representaion_BPSK = cos(2*pi*fc*t_1bitperiod);
    else
        bit_representaion_BPSK = cos(2*pi*fc*t_1bitperiod + pi);
    end
    mod_BPSK = [mod_BPSK bit_representaion_BPSK];
end

figure (20)
grid on
plot(t_bitstream, mod_BPSK, 'LineWidth', 1.5);
xlabel('Time[s]')
ylabel('Amplitude[V]')
title('BPSK Modulated Signal')

% QPSK Modulation
t_2_bit_period = Tb/nb:Tb/nb:2*Tb; % The length of 2 bits is 0.5sec (Tb)
mod_QPSK = [];
for i = 1:2:bit_string_len-1
    if random_bit_string(i) == 0
        if random_bit_string(i+1) == 0
            bit_representaion_QPSK = cos(2*pi*fc*t_2_bit_period);
        else
            bit_representaion_QPSK = sin(2*pi*fc*t_2_bit_period);
        end
    else 
        if random_bit_string(i+1) == 0
            bit_representaion_QPSK = -cos(2*pi*fc*t_2_bit_period);            
        else
            bit_representaion_QPSK = -sin(2*pi*fc*t_2_bit_period);
        end
    end
    mod_QPSK = [mod_QPSK bit_representaion_QPSK];
end

figure (21)
grid on
plot(t_bitstream, mod_QPSK, 'LineWidth', 1.5);
xlabel('Time[s]')
ylabel('Amplitude[V]')
title('QPSK Modulated Signal')

% 8-PSK Modulation
t_3_bit_period = Tb/nb:Tb/nb:3*Tb; % The length of 2 bits is 0.75sec (3Tb)
mod_PSK_8 = [];
for i = 1:3:bit_string_len-2
    if random_bit_string(i) == 0
        if random_bit_string(i+1) == 0
            if random_bit_string(i+2) == 0
                bit_representaion_8PSK = cos(2*pi*fc*t_3_bit_period);
            else
                bit_representaion_8PSK = cos(2*pi*fc*t_3_bit_period + pi/4);
            end
        else
            if random_bit_string(i+2) == 0
                bit_representaion_8PSK = cos(2*pi*fc*t_3_bit_period + 3*pi/4);
            else   
                bit_representaion_8PSK = cos(2*pi*fc*t_3_bit_period + pi/2);
            end
        end
    else 
        if random_bit_string(i+1) == 0
            if random_bit_string(i+2) == 0
                bit_representaion_8PSK = cos(2*pi*fc*t_3_bit_period + 7*pi/4);
            else
                bit_representaion_8PSK = cos(2*pi*fc*t_3_bit_period + 3*pi/2);
            end
        else
            if random_bit_string(i+2) == 0
                bit_representaion_8PSK = cos(2*pi*fc*t_3_bit_period + pi);
            else
                bit_representaion_8PSK = cos(2*pi*fc*t_3_bit_period + 5*pi/4);
            end
        end
    end
    mod_PSK_8 = [mod_PSK_8 bit_representaion_8PSK];
end

figure (22)
grid on
plot(t_bitstream, mod_PSK_8, 'LineWidth', 1.5);
xlabel('Time[s]')
ylabel('Amplitude[V]')
title('8-PSK Modulated Signal')


% 3.b)

A_3_b = 3;
%A_3_b = 9;

B_PAM_random_bit_string = [];
for i = 1:bit_string_len
    if random_bit_string(i) == 1
        B_PAM_bit_signal = A_3_b*ones(1, nb);
    else 
        B_PAM_bit_signal = -A_3_b*ones(1, nb);
    end
    B_PAM_random_bit_string = [B_PAM_random_bit_string B_PAM_bit_signal];
end

%t_bitstream = Tb/nb:Tb/nb:bs_len*Tb;
figure (23)
plot(t_bitstream, B_PAM_random_bit_string, 'LineWidth', 2.5);
grid on
xlabel('Time[s]')
ylim([(-A_3_b -1) (A_3_b +1)])
ylabel('Amplitude[V]')
title('B-PAM of Random Bit String')  


% 3.c)

scatterplot(0.5*B_PAM_random_bit_string,1, 0, 'b*'); 
title("B-PAM Constellation Diagram")
grid on;


% 3.d)

AWGN_3_d_5db = awgn(B_PAM_random_bit_string, 5);
figure (25)
plot(t_bitstream, AWGN_3_d_5db)
grid on
xlabel('Time [s]')
ylabel('Amplitude [V]')
title('AWGN for E_b/N_0 = 5dB')

AWGN_3_d_15db = awgn(B_PAM_random_bit_string, 15);
figure (26)
plot(t_bitstream, AWGN_3_d_15db)
grid on
xlabel('Time [s]')
ylabel('Amplitude [V]')
title('AWGN for E_b/N_0 = 15dB')

% 3.e)

bpam_modulated = pammod(random_bit_string, 2);

bpam_modulated_noisy_5dB = awgn(bpam_modulated, 8);
scatterplot(A_3_b*0.5* bpam_modulated_noisy_5dB);
title('B-PAM Noisy with E_b/N_0 of 5dB')
grid on

bpam_modulated_noisy_15dB = awgn(bpam_modulated, 18);
scatterplot(A_3_b*0.5*bpam_modulated_noisy_15dB);
title('B-PAM Noisy with E_b/N_0 of 15dB')
grid on


% 3.st)

rng('default') % Set random number seed for repeatability
M = 2;
EbNo = 0:15;
[BER] = berawgn(EbNo,'pam',M);
figure(29);
semilogy(EbNo,BER,'r');
legend('Theoretical BER');
title('Theoretical Error Rate');
xlabel('E_b/N_0 (dB)');
ylabel('Bit Error Rate');
grid on;

n_bits = 1000000; % Number of random bits to process
k_bits = log2(M); % Number of bits per symbol
snr_random_bits = EbNo+3+10*log10(k_bits); % In dB
y_with_noise = zeros(n_bits,length(snr_random_bits));
z_snr = zeros(n_bits,length(snr_random_bits));
errVec = zeros(3,length(EbNo));


errcalc = comm.ErrorRate;


x_random_bits = randi([0 1],[1000000 , 1]); % Create message signal
y_random_bits_pam = pammod(x_random_bits,M); % Modulate

for jj = 1:length(snr_random_bits)
    reset(errcalc)
    y_with_noise(:,jj) = awgn(real(y_random_bits_pam),snr_random_bits(jj),'measured'); % Add AWGN
    z_snr(:,jj) = pamdemod(complex(y_with_noise(:,jj)),M); % Demodulate
    errVec(:,jj) = errcalc(x_random_bits,z_snr(:,jj)); % Compute SER from simulation
end


hold on;
semilogy(EbNo,errVec(1,:),'b.');
legend('Theoretical BER','Empirical BER');
title('Comparison of Theoretical and Empirical Error Rates');
hold off;

%-------------------------------------------------------------------------------

% ######################## EXERCISE 4 ########################


% 4.a)

QPSK_4 = comm.QPSKModulator(BitInput=true);
QPSK_bits_4 = A_3_b*QPSK_4(random_bit_string);
scatterplot(0.5*QPSK_bits_4, 1, 0, 'bx');
grid on;
title('QPSK Constellation diagram')
axis([-6 6 -6 6])

% Gray Code
text(0.5*A_3_b/sqrt(2)-0.12, 0.5*A_3_b/sqrt(2)-0.22, '00', 'Color', [1 0 0]);
text(-0.5*A_3_b/sqrt(2)-0.12, 0.5*A_3_b/sqrt(2)-0.22, '01', 'Color', [1 0 0]);
text(-0.5*A_3_b/sqrt(2)-0.12, -0.5*A_3_b/sqrt(2)+0.22, '11', 'Color', [1 0 0]);
text(0.5*A_3_b/sqrt(2)-0.12, -0.5*A_3_b/sqrt(2)+0.22, '10', 'Color', [1 0 0]);


% 4.b)

AWGN_4_b_5dB = awgn(QPSK_bits_4, 5);
scatterplot(0.5*AWGN_4_b_5dB, 1, 0, 'b*');
title('QPSK Constellation - AWGN - E_b/N_0 = 5 dB')
grid on

AWGN_4_b_15dB = awgn(QPSK_bits_4, 15);
scatterplot(0.5*AWGN_4_b_15dB, 1, 0, 'b*');
title('QPSK Constellation - AWGN - E_b/N_0 = 15 dB')
grid on


% 4.c)

M_qpsk = 4;
EbNo = 0:15;
[BER] = berawgn(EbNo,'psk',M_qpsk ,'diff');
figure(33);
semilogy(EbNo,BER,'b');
title('Comparing BER for QPSK and BPSK modulation');
xlabel('E_b/N_0 (dB)');
ylabel('Bit Error Rate');
grid on;

M_bpsk = 2;
n_bits_bpsk = 1000000; % Number of random bits to process
k_bits_bpsk = log2(M_bpsk); % Number of bits per symbol
snr_random_bits_bpsk = EbNo+10*log10(k_bits_bpsk); % In dB
y_with_noise_bpsk = zeros(n_bits_bpsk,length(snr_random_bits_bpsk));
z_snr_bpsk = zeros(n_bits_bpsk,length(snr_random_bits_bpsk));
errVec_bpsk = zeros(3,length(EbNo));


errcalc = comm.ErrorRate;


x_random_bits_bpsk = randi([0 1],1000000 , 1); % Create message signal
y_random_bits_bpsk = pskmod(x_random_bits_bpsk, M_bpsk); % Modulate

for jj = 1:length(snr_random_bits_bpsk)
    reset(errcalc)
    y_with_noise_bpsk(:,jj) = awgn(y_random_bits_bpsk,snr_random_bits_bpsk(jj),'measured'); % Add AWGN
    z_snr_bpsk(:,jj) = pskdemod(complex(y_with_noise_bpsk(:,jj)), M_bpsk); % Demodulate
    errVec_bpsk(:,jj) = errcalc(x_random_bits_bpsk,z_snr_bpsk(:,jj)); % Compute SER from simulation
end


hold on;
semilogy(EbNo,errVec_bpsk(1,:),'g.');

M_qpsk = 4;
n_bits_qpsk = 1000000; % Number of random bits to process
k_bits_qpsk = log2(M_qpsk); % Number of bits per symbol
snr_random_bits_qpsk = EbNo+10*log10(k_bits_qpsk); % In dB
y_with_noise_qpsk = zeros(n_bits_qpsk,length(snr_random_bits_qpsk));
z_snr_qpsk = zeros(n_bits_qpsk,length(snr_random_bits_qpsk));
errVec_qpsk = zeros(3,length(EbNo));


errcalc = comm.ErrorRate;


x_random_bits_qpsk = randi([0 1],1000000 , 1); % Create message signal
y_random_bits_qpsk = pskmod(x_random_bits_qpsk, M_qpsk); % Modulate
for jj = 1:length(snr_random_bits_qpsk)
    reset(errcalc)
    y_with_noise_qpsk(:,jj) = awgn(y_random_bits_qpsk,snr_random_bits_qpsk(jj),'measured'); % Add AWGN
    z_snr_qpsk(:,jj) = pskdemod(complex(y_with_noise_qpsk(:,jj)), M_qpsk); % Demodulate
    errVec_qpsk(:,jj) = errcalc(x_random_bits_qpsk,z_snr_qpsk(:,jj)); % Compute SER from simulation
end



semilogy(EbNo,errVec_qpsk(1,:),'r.');
legend('Theoretical BER','BPSK Empirical BER','QPSK Empirical BER');
hold off ;


% 4.d)

% 4.d (i)

file_id = fopen('rice_odd.txt', 'r');
file_text = fscanf(file_id, '%c');
binary_text = dec2bin(file_text);

% 4.d (ii)

binary_text_dec = bin2dec(binary_text);
N_text=8; %bits
L_text = 2^N_text ; %number of quantized levels
D_text = (max(binary_text_dec) - min(binary_text_dec))/L_text; %step size
I_text = round((binary_text_dec - min(binary_text_dec))/D_text) ; %index
txt_quantized = min(binary_text_dec) + I_text*D_text;
figure(34);
n_txt = 0:1:484;
stem (n_txt,txt_quantized,'rx-');
title('Text file')
xlabel('Characters')
ylabel('ASCII CODE')


% 4.d (iii)

txt_quantized_bits = dec2bin(round(txt_quantized));
txt_quantized_bit_stream = zeros(3396, 1);
index_bit = 1;

for i = 1:485
    for j = 1:7
        if txt_quantized_bits(i, j) == '1'
            txt_quantized_bit_stream(index_bit) = 1;
        end
        index_bit = index_bit + 1;
    end
end

text_QPSK_mod = QPSK_4(txt_quantized_bit_stream);

scatterplot(text_QPSK_mod);
title("QPSK constellation diagram for modulated bit text")
axis([-2 2 -2 2])

% 4.d (iv)

text_QPSK_mod_noisy5dB = awgn(text_QPSK_mod, 5);
text_QPSK_mod_noisy15dB = awgn(text_QPSK_mod, 15);

scatterplot(text_QPSK_mod_noisy5dB);
title("Text QPSK with E_s/N_0 = 5dB")
scatterplot(text_QPSK_mod_noisy15dB);
title("Text QPSK with E_s/N_0 = 15dB")

% 4.d (v)

% Demodulator the returns singal in bit representation
text_QPSK_demod = comm.QPSKDemodulator(BitOutput=true); 

% Demodulating the noisy signal
text_QPSK_demod_5dB_dec = text_QPSK_demod(text_QPSK_mod_noisy5dB);
text_QPSK_demod_15dB_dec = text_QPSK_demod(text_QPSK_mod_noisy15dB);

% 4.d (vi)

[num_QPSK_demod_5dB, BER_QPSK_demod_5dB] = symerr(txt_quantized_bit_stream, text_QPSK_demod_5dB_dec);
[num_QPSK_demod_15dB, BER_QPSK_demod_15dB] = symerr(txt_quantized_bit_stream, text_QPSK_demod_15dB_dec);

BER_QPSK_demod_5dB_Theoretical = qfunc(sqrt(10^(5/10)));
BER_QPSK_demod_15dB_Theoretical = qfunc(sqrt(10^(15/10)));

% 4.d (vii)

demod_file1  = char(bin2dec(reshape(char('0' + text_QPSK_demod_15dB_dec(1:3395) ),7,[]).'))';
demod_file2  = char(bin2dec(reshape(char('0' + text_QPSK_demod_5dB_dec(1:3395) ),7,[]).'))';


%saving in text file (.txt)
demoded_file_5dB = fopen('demoded_text_5dB.txt', 'w');
fprintf(demoded_file_5dB, '%c', demod_file2);
demoded_file_15dB = fopen('demoded_text_15dB.txt', 'w');
fprintf(demoded_file_15dB, '%c', demod_file1);


%-------------------------------------------------------------------

% ######################## EXERCISE 5 ########################

%5.a)

[sound,fs_sound] = audioread('soundfile1_lab2.wav'); 
dt_sound = 1/fs_sound;
t_sound = 0:dt_sound:(length(sound)*dt_sound)-dt_sound; %time(in seconds)
figure(38);
plot(t_sound,sound);
 

%5.b)

sound_bits = 8; %bits
L_sound = 2^sound_bits ; %number of quantized levels
D_sound = (max(sound) - min(sound))/L_sound; %step size
I_sound = round((sound - min(sound))/D_sound) ; %index
quantized_sound = min(sound) + I_sound*D_sound;
figure(39);
stem (t_sound,quantized_sound,'rx-');
xlabel('Time[s]');
ylabel('Amplitude');
title('Quantized sound signal at 8 bits');

%5.c)

A_s = quantized_sound*1000;

quantized_sound_bits = dec2bin(A_s,16)-'0';

quantized_sound_bit_stream = reshape(quantized_sound_bits.',[],1);
sound_qpskmod = comm.QPSKModulator('BitInput',true);
sound_QPSK_mod = sound_qpskmod(quantized_sound_bit_stream);

scatterplot(sound_QPSK_mod);
title("QPSK Constellation diagram for modulated audio file")

%5.d)

sound_QPSK_mod_noisy4dB = awgn(sound_QPSK_mod, 4);
sound_QPSK_mod_noisy14dB = awgn(sound_QPSK_mod, 14);

scatterplot(sound_QPSK_mod_noisy4dB);
title("audio QPSK with E_s/N_0 = 4dB")
scatterplot(sound_QPSK_mod_noisy14dB);
title("audio QPSK with E_s/N_0 = 14dB")

%5.e)

qpskdemod_sound = comm.QPSKDemodulator('BitOutput',true); 
sound_QPSK_demod_4dB_dec = qpskdemod_sound(sound_QPSK_mod_noisy4dB);
sound_QPSK_demod_14dB_dec = qpskdemod_sound(sound_QPSK_mod_noisy14dB);

%5.f)

[num_QPSK_demod_4db, BER_QPSK_demod_4db] = symerr(quantized_sound_bit_stream, sound_QPSK_demod_4dB_dec);
[num_QPSK_demod_14db, BER_QPSK_demod_14db] = symerr(quantized_sound_bit_stream, sound_QPSK_demod_14dB_dec);

BER_QPSK_demod_4dB_Theoretical = qfunc(sqrt(10^(4/10)));
BER_QPSK_demod_14dB_Theoretical = qfunc(sqrt(10^(14/10)));

%5.g)

new_sound_1 = char('0'+ (reshape(sound_QPSK_demod_14dB_dec, 16, []).'));
new_sound_14dB = typecast(uint16(bin2dec(new_sound_1)),'int16');
regained_sound1 = double(abs(new_sound_14dB))./1000;

new_sound_2 = char('0'+ (reshape(sound_QPSK_demod_4dB_dec, 16, []).'));
new_sound_4dB = typecast(uint16(bin2dec(new_sound_2)),'int16');
regained_sound2 = double(abs(new_sound_4dB))./1000;

regained_sound1_norm = regained_sound1 / max(regained_sound1);
regained_sound2_norm = regained_sound2 / max(regained_sound2);

audiowrite("regained_sound_14dB.wav", regained_sound1_norm , 44100);
audiowrite("regained_sound_4dB.wav", regained_sound2_norm, 44100);


%###################################################################
%###################################################################
%###################################################################

% Code by Litsos Ioannis: 
%{

%%%%%%%%%%%%%%%%%%%%%%%%% Exercise 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%1.a

%Triangular pulse
fm = 9000;
T = 4*(1/fm);
fs = 10000000;
t0 = 0:1/fs:T-1/fs;
x_triangle = 4*(sawtooth(2*pi*fm*t0,1/2));
figure (1);
plot(t0,x_triangle);
xlabel('Time[s]');
ylabel('Amplitude[V]');
title ('Triangular Pulse at fm = 9kHz');
grid on

%(i)
fs_1=30*fm; 
dt1=1/fs_1;  
t1=0:dt1:T-1/fs;
x_triangle_1 = 4*(sawtooth(2*pi*fm*t1,1/2));
figure (2);
stem(t1,x_triangle_1,'rx-');
xlabel('Time [s]'); ylabel('Amplitude[V]');
title('Sampling the triangular pulse at 30*fm')

%(ii)
fs_2=50*fm; 
dt2=1/fs_2;
t2=0:dt2:T-1/fs;
x_triangle_2 = 4*(sawtooth(2*pi*fm*t2,1/2));
figure (3);
stem(t2,x_triangle_2,'rx-');
xlabel('Time [s]'); ylabel('Amplitude[V]');
title('Sampling the triangular pulse at 50*fm');

%(iii)
figure (4);
hold on;
stem(t1,x_triangle_1,'bo-');
stem(t2,x_triangle_2,'rx-');
title('Samples from both sampling frequencies');
legend ('Sampling at 30*fm' , 'Sampling at 50*fm');
hold off;

%1.b
fs_3 = 4*fm ; 
dts=1/fs_3;  
figure (5);
ts=0:dts:T-1/fs;
x_triangle_s = 4*(sawtooth(2*pi*fm*ts,1/2));
stem(ts,x_triangle_s,'rx-');
xlabel('Time [s]'); ylabel('Amplitude[V]');
title('Sampling the triangular pulse at 4*fm');

% DFT of the triangular pulse 
Y = fft(x_triangle);
N=length(Y);
f_y = 0:fs/N:fs/2-fs/N;
figure (6)
plot(f_y, abs(Y(1:N/2))); % DFT is symmetrical, we plot only half of the transform
xlabel('Frequency (Hz)');
ylabel('Amplitude');
title('Fourier spectrum of the triangular pulse');

%1.c

 %(i){

  %sinusoid function
  z_t = sin(2*pi*fm*t0);
  figure (7);
  plot(t0,z_t); 
  xlabel('Time[s]');
  ylabel('Amplitude[V]');
  title('Sinusoid function');

    %(a){
     %(i)
      figure (8);
      z1 = sin(2*pi*fm*t1);
      stem(t1,z1,'rx-');
      xlabel('Time [s]'); ylabel('Amplitude[V]');
      title('Sampling of the sinusoid function at sampling rate 30*fm');

     %(ii)
      z2 = sin(2*pi*fm*t2);
      figure (9);
      stem(t2,z2,'bo-');
      xlabel('Time [s]'); ylabel('Amplitude[V]');
      title('Sampling of the sinusoid function at sampling rate 50*fm');

     %(iii)
      figure (10);
      hold on;
      stem(t1,z1,'rx-');
      stem(t2,z2,'bo-');
      hold off;
      xlabel('Time [s]'); ylabel('Amplitude[V]');
      title('Samples from both sampling rates');
      legend ('Sampling at 30*fm' ,'Sampling at 50*fm');
       
   %(b) {
    z_s = sin(2*pi*fm*ts);
    figure(11);
    stem(ts,z_s,'rx-');
    xlabel('Time [s]'); ylabel('Amplitude[V]');
    title('Sampling frequency 4*fm(Sinusoidal function)');
   %}   }

%(ii){
 l = 1000;
 fm_total = gcd(fm, fm + l);
 T_q = 1/fm_total ;
 t_q = 0:1/fs:T_q-1/fs;
 z_q = sin(2*pi*fm*t_q);
 q = z_q + sin(2*pi*(fm + l)*t_q);
 figure(50);
 plot(t_q,q);
 xlabel('Time[s]');
 ylabel('Amplitude[V]');
 title('Function q(t)= z(t) + Asin(2π(fm + Λ)t)');

  %(a) {
    %(i)
      figure (12);
      dt1_q=1/fs_1;  
      t1_q=0:dt1_q:T_q-1/fs;
      z1_q = sin(2*pi*fm*t1_q);
      q1 = z1_q + sin(2*pi*(fm + l)*t1_q);
      stem(t1_q,q1,'rx-');
      xlabel('Time [s]'); ylabel('Amplitude[V]');
      title('Sampling of the sinusoid function q(t) at sampling rate 30*fm');
     
     %(ii)
      figure(13);
      dt2_q=1/fs_2;
      t2_q=0:dt2_q:T_q-1/fs;
      z2_q = sin(2*pi*fm*t2_q);
      q2 = z2_q + sin(2*pi*(fm + l)*t2_q);
      stem(t2_q,q2,'bo-');
      xlabel('Time [s]'); ylabel('Amplitude[V]');
      title('Sampling of the sinusoid function q(t)at sampling rate 50*fm');
      
     %(iii)
      figure (14);
      hold on;
      stem(t1_q,q1,'rx-');
      stem(t2_q,q2,'bo-');
      hold off;
      xlabel('Time [s]'); ylabel('Amplitude[V]');
      title('Samples of q(t) from both sampling rates');
      legend ('Sampling rate 30*fm' , 'Sampling rate 50*fm');
           
 %(b) {
    fs_3 = 4*fm ; 
    dts=1/fs_3;  
    figure (15);
    ts_q=0:dts:T_q-1/fs;
    z_s = sin(2*pi*fm*ts_q);
    q_s = z_s + sin(2*pi*(fm + l)*ts_q);
    stem(ts_q,q_s,'rx-');
    xlabel('Time [s]'); ylabel('Amplitude[V]');
    title('Sampling of the sinusoid function q(t) at sampling rate 4*fm');
   %}  } }
          
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EXERCISE 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
 
 
%2.a
   Max_Value_Dec = 32;
   dec_sequence = (0:Max_Value_Dec-1);
   gray_sequence = qammod(dec_sequence, Max_Value_Dec, 'gray');

   [gray_sec,map_gray] = bin2gray(dec_sequence,'qam',Max_Value_Dec);

   gray_code = dec2bin(gray_sec);

   N=5; %bits
   L = 2^N ; %number of quantized levels
   D = (max(x_triangle_1) - min(x_triangle_1))/L; %step size
   I = round((x_triangle_1 - min(x_triangle_1))/D) ; %index
   x_quantized = min(x_triangle_1) + I*D;
   figure(16);
   stem (t1,x_quantized,'rx-');
   xlabel('Time[s]');
   ylabel('Gray Code');
   title('Quantized triangular pulse at 5 bits after sampling it at 30*fm');
   yticks([-4 -3.75 -3.5 -3.25 -3 -2.75 -2.5 -2.25 -2 -1.75 -1.5 -1.25 -1 -0.75 -0.5 -0.25 0.25 0.5 0.75 1 1.25 1.5 1.75 2 2.25 2.5 2.75 3 3.25 3.5 3.75 4])
   yticklabels(gray_code);

 
 %2.b
 %(i)
  z = x_triangle_1 - x_quantized ; 
  error_first10 = z(1, 1:10);
  std_10 = sqrt(var(error_first10));
  
 %(ii)
  error_first20 = z(1, 1:20);
  std_20 = sqrt(var(error_first20));
 
  %(iii)
  %SNR for the first 10 samples of our signal
  numerator = 3*2^(2*N);
  rms_first10 = rms(x_triangle_1(1:10));
  f1 = max(x_triangle_1(1:10))/rms_first10;
  snr_first10 = 10*log10(numerator./(f1^2));
  
  %SNR for the first 20 samples of our signal
  rms_first20 = rms(x_triangle_1(1:20));
  f2 = max(x_triangle_1(1:20))/rms_first20;
  snr_first20 = 10*log10(numerator./(f2^2));
  
  %SNR of the whole signal
  rms_x_triangle_1 = rms(x_triangle_1);
  f_x_triangle_1 = max(x_triangle_1)/rms_x_triangle_1;
  snr_theoretical = 10*log10(numerator./(f_x_triangle_1^2));
  
  
  
  
  %2.c
  bit_stream = char(zeros(30,5));
  for i = 1:30
      if x_quantized(i) < 0
        index_gc = ( x_quantized(i) + 4 ) / 0.25 + 1;
        bit_stream(i,:) = gray_code(index_gc,:);
      else 
        index_gc = ( x_quantized(i) + 4 ) / 0.25 ;
        bit_stream(i,:) = gray_code(index_gc,:);
      end
  end
  
 x_axis_bit_stream = [];
 y_axis_bit_stream = [];
 index_bs = 1;
 for i = 1:30
    for j = 1:5
        x_axis_bit_stream = [x_axis_bit_stream 2*(index_bs-1) 2*(index_bs-1+0.5) 2*(index_bs-1+0.5) 2*(index_bs)];
        if(bit_stream(i,j) == '0')
            y_axis_bit_stream = [y_axis_bit_stream -9 -9 0 0];
        else 
            y_axis_bit_stream = [y_axis_bit_stream 9 9 0 0];
        end
        index_bs = index_bs + 1;
    end
end

figure (17)
plot(x_axis_bit_stream, y_axis_bit_stream), axis([0, 300 , -10, 10]);
title('Line coding Polar RZ of the quantized triangular pulse');
xlabel ('Time[msec]');
ylabel ('Amplitude(V)');

%%%%%%%%%%%%%%%%%%%%%%%%%% EXERCISE 3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % random 36 bit string
random_bit_string = randi([0,1],[36,1]);


bs_len = length(random_bit_string); % Length of Bit String
Tb = 0.25;   % Bit period
fc = 2;   % Carrier frequency
nb = 100; % Digital Signal per Bit

% Plotting the Random Digital String
rand_bs_digital_string = [];
for i = 1:bs_len
    if random_bit_string(i) == 1
        bit_signal = ones(1, nb);
    else 
        bit_signal = zeros(1, nb);
    end
    rand_bs_digital_string = [rand_bs_digital_string bit_signal];
end

t_bitstream = Tb/nb:Tb/nb:bs_len*Tb;
figure (18)
grid on
plot(t_bitstream, rand_bs_digital_string, 'LineWidth', 2.5);
xlabel('Time[s]')
ylim([-0.5 1.5])
ylabel('Amplitude[V]')
title('Digital Signal of Random Bit String')  


% 3.a

% BPSK Modulation
t_1bitperiod = Tb/nb:Tb/nb:Tb; % The length of 1 bit is 0.25sec (Tb)
BPSK_mod = [];
for i = 1:bs_len
    if random_bit_string(i) == 1
        bit_representaion_BPSK = cos(2*pi*fc*t_1bitperiod);
    else
        bit_representaion_BPSK = cos(2*pi*fc*t_1bitperiod + pi);
    end
    BPSK_mod = [BPSK_mod bit_representaion_BPSK];
end

figure (19)
grid on
plot(t_bitstream, BPSK_mod, 'LineWidth', 1.5);
xlabel('Time[s]')
ylabel('Amplitude[V]')
title('BPSK Modulated Signal')
% QPSK Modulation
t_2bitperiod = Tb/nb:Tb/nb:2*Tb; % The length of 2 bits is 0.5sec (Tb)
QPSK_mod = [];
for i = 1:2:bs_len-1
    if random_bit_string(i) == 0
        if random_bit_string(i+1) == 0
            bit_representaion_QPSK = cos(2*pi*fc*t_2bitperiod);
        else
            bit_representaion_QPSK = sin(2*pi*fc*t_2bitperiod);
        end
    else 
        if random_bit_string(i+1) == 0
            bit_representaion_QPSK = -cos(2*pi*fc*t_2bitperiod);            
        else
            bit_representaion_QPSK = -sin(2*pi*fc*t_2bitperiod);
        end
    end
    QPSK_mod = [QPSK_mod bit_representaion_QPSK];
end

figure (20)
grid on
plot(t_bitstream, QPSK_mod, 'LineWidth', 1.5);
xlabel('Time[s]')
ylabel('Amplitude[V]')
title('QPSK Modulated Signal')
% 8-PSK Modulation
t_3bitperiod = Tb/nb:Tb/nb:3*Tb; % The length of 2 bits is 0.75sec (3Tb)
PSK8_mod = [];
for i = 1:3:bs_len-2
    if random_bit_string(i) == 0
        if random_bit_string(i+1) == 0
            if random_bit_string(i+2) == 0
                bit_representaion_8PSK = cos(2*pi*fc*t_3bitperiod);
            else
                bit_representaion_8PSK = cos(2*pi*fc*t_3bitperiod + pi/4);
            end
        else
            if random_bit_string(i+2) == 0
                bit_representaion_8PSK = cos(2*pi*fc*t_3bitperiod + 3*pi/4);
            else   
                bit_representaion_8PSK = cos(2*pi*fc*t_3bitperiod + pi/2);
            end
        end
    else 
        if random_bit_string(i+1) == 0
            if random_bit_string(i+2) == 0
                bit_representaion_8PSK = cos(2*pi*fc*t_3bitperiod + 7*pi/4);
            else
                bit_representaion_8PSK = cos(2*pi*fc*t_3bitperiod + 3*pi/2);
            end
        else
            if random_bit_string(i+2) == 0
                bit_representaion_8PSK = cos(2*pi*fc*t_3bitperiod + pi);
            else
                bit_representaion_8PSK = cos(2*pi*fc*t_3bitperiod + 5*pi/4);
            end
        end
    end
    PSK8_mod = [PSK8_mod bit_representaion_8PSK];
end

figure (21)
grid on
plot(t_bitstream, PSK8_mod, 'LineWidth', 1.5);
xlabel('Time[s]')
ylabel('Amplitude[V]')
title('8-PSK Modulated Signal')
  % 3.b

B_PAM_rand_bs = [];
for i = 1:bs_len
    if random_bit_string(i) == 1
        B_PAM_bit_signal = 9*ones(1, nb);
    else 
        B_PAM_bit_signal = -9*ones(1, nb);
    end
    B_PAM_rand_bs = [B_PAM_rand_bs B_PAM_bit_signal];
end

%t_bitstream = Tb/nb:Tb/nb:bs_len*Tb;
figure (22)
plot(t_bitstream, B_PAM_rand_bs, 'LineWidth', 2.5);
grid on
xlabel('Time[s]')
ylim([-10 10])
ylabel('Amplitude[V]')
title('B-PAM of Random Bit String')  

 %3.c
 grid on
 scatterplot(B_PAM_rand_bs/2, 1,0,'b*');
 title ('Constellation of B-PAM');
 
 
 % 3.d
awgn_3d_5db = awgn(B_PAM_rand_bs, 5);
figure (24)
plot(t_bitstream, awgn_3d_5db)
grid on
xlabel('Time [s]')
ylabel('Amplitude [V]')
title('AWGN for E_b/N_0 = 5db')

awgn_3d_15db = awgn(B_PAM_rand_bs, 15);
figure (25)
plot(t_bitstream, awgn_3d_15db)
grid on
xlabel('Time [s]')
ylabel('Amplitude [V]')
title('AWGN for E_b/N_0 = 15db')

 %3.e
bpam_modulated = pammod(random_bit_string, 2);
bpam_modulated_noisy_5dB = awgn(bpam_modulated, 8);
grid on;
scatterplot(0.5*9*bpam_modulated_noisy_5dB);
title('BPAM Noisy with Eb/N0 of 5dB')
bpam_modulated_noisy_15dB = awgn(bpam_modulated, 18);
grid on;
scatterplot(0.5*9*bpam_modulated_noisy_15dB);
title('BPAM Noisy with Eb/N0 of 15dB');
 

% 3.st
M = 2;
EbNo = 0:15;
[BER] = berawgn(EbNo,'pam',M);
figure(28);
semilogy(EbNo,BER,'r');
legend('Theoretical BER');
title('Theoretical Error Rate');
xlabel('E_b/N_0 (dB)');
ylabel('Bit Error Rate');
grid on;

n_bits = 1000000; % Number of random bits to process
k_bits = log2(M); % Number of bits per symbol
snr_random_bits = EbNo+3+10*log10(k_bits); % In dB
y_with_noise = zeros(n_bits,length(snr_random_bits));
z_snr = zeros(n_bits,length(snr_random_bits));
errVec = zeros(3,length(EbNo));


errcalc = comm.ErrorRate;


x_random_bits = randi([0 1],[1000000 , 1]); % Create message signal
y_random_bits_pam = pammod(x_random_bits,M); % Modulate

for jj = 1:length(snr_random_bits)
    reset(errcalc)
    y_with_noise(:,jj) = awgn(real(y_random_bits_pam),snr_random_bits(jj),'measured'); % Add AWGN
    z_snr(:,jj) = pamdemod(complex(y_with_noise(:,jj)),M); % Demodulate
    errVec(:,jj) = errcalc(x_random_bits,z_snr(:,jj)); % Compute SER from simulation
end


hold on;
semilogy(EbNo,errVec(1,:),'b.');
legend('Theoretical BER','Empirical BER');
title('Comparison of Theoretical and Empirical Error Rates');
hold off;

 
 
 %%%%%%%%%%%%%%%%%%%%%% EXERCISE 4 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
 %4.a
    qpskmod = comm.QPSKModulator('BitInput',true);
    qpsk_bits_4 = 9*qpskmod(random_bit_string);
    scatterplot(0.5*qpsk_bits_4,1,0,'b*');
    axis([-8 8 -8 8]);
    title('QPSK Modulated Signal')
    text(0.5*(9/sqrt(2))-0.12, 0.5*(9/sqrt(2))-0.22, '00', 'Color', [1 0 0]);
    text(-0.5*(9/sqrt(2))-0.12, 0.5*(9/sqrt(2))-0.22, '01', 'Color', [1 0 0]);
    text(-0.5*(9/sqrt(2))-0.12, -0.5*(9/sqrt(2))+0.22, '11', 'Color', [1 0 0]);
    text(0.5*(9/sqrt(2))-0.12, -0.5*(9/sqrt(2))+0.22, '10', 'Color', [1 0 0]);

%4.b
    awgn_4b_5db = awgn(qpsk_bits_4, 5);
    scatterplot(0.5*awgn_4b_5db, 1, 0, 'b*');
    title('Constellation of AWGN noise for Eb/No = 5dB');
    grid on

    awgn_4b_15db = awgn(qpsk_bits_4, 15);
    scatterplot(0.5*awgn_4b_15db, 1, 0, 'b*');
    title ('Constellation of AWGN noise for Eb/No = 15dB');
    grid on

% 4.c
M_qpsk = 4;
EbNo = 0:15;
[BER] = berawgn(EbNo,'psk',M_qpsk ,'diff');
figure(32);
semilogy(EbNo,BER,'b');
title('Comparing BER for QPSK and BPSK modulation');
xlabel('E_b/N_0 (dB)');
ylabel('Bit Error Rate');
grid on;

M_bpsk = 2;
n_bits_bpsk = 1000000; % Number of random bits to process
k_bits_bpsk = log2(M_bpsk); % Number of bits per symbol
snr_random_bits_bpsk = EbNo+10*log10(k_bits_bpsk); % In dB
y_with_noise_bpsk = zeros(n_bits_bpsk,length(snr_random_bits_bpsk));
z_snr_bpsk = zeros(n_bits_bpsk,length(snr_random_bits_bpsk));
errVec_bpsk = zeros(3,length(EbNo));


errcalc = comm.ErrorRate;


x_random_bits_bpsk = randi([0 1],1000000 , 1); % Create message signal
y_random_bits_bpsk = pskmod(x_random_bits_bpsk, M_bpsk); % Modulate

for jj = 1:length(snr_random_bits_bpsk)
    reset(errcalc)
    y_with_noise_bpsk(:,jj) = awgn(y_random_bits_bpsk,snr_random_bits_bpsk(jj),'measured'); % Add AWGN
    z_snr_bpsk(:,jj) = pskdemod(complex(y_with_noise_bpsk(:,jj)), M_bpsk); % Demodulate
    errVec_bpsk(:,jj) = errcalc(x_random_bits_bpsk,z_snr_bpsk(:,jj)); % Compute SER from simulation
end


hold on;
semilogy(EbNo,errVec_bpsk(1,:),'g.');

M_qpsk = 4;
n_bits_qpsk = 1000000; % Number of random bits to process
k_bits_qpsk = log2(M_qpsk); % Number of bits per symbol
snr_random_bits_qpsk = EbNo+10*log10(k_bits_qpsk); % In dB
y_with_noise_qpsk = zeros(n_bits_qpsk,length(snr_random_bits_qpsk));
z_snr_qpsk = zeros(n_bits_qpsk,length(snr_random_bits_qpsk));
errVec_qpsk = zeros(3,length(EbNo));


errcalc = comm.ErrorRate;


x_random_bits_qpsk = randi([0 1],1000000 , 1); % Create message signal
y_random_bits_qpsk = pskmod(x_random_bits_qpsk, M_qpsk); % Modulate

for jj = 1:length(snr_random_bits_qpsk)
    reset(errcalc)
    y_with_noise_qpsk(:,jj) = awgn(y_random_bits_qpsk,snr_random_bits_qpsk(jj),'measured'); % Add AWGN
    z_snr_qpsk(:,jj) = pskdemod(complex(y_with_noise_qpsk(:,jj)), M_qpsk); % Demodulate
    errVec_qpsk(:,jj) = errcalc(x_random_bits_qpsk,z_snr_qpsk(:,jj)); % Compute SER from simulation
end



semilogy(EbNo,errVec_qpsk(1,:),'r.');
legend('Theoretical BER','BPSK Empirical BER','QPSK Empirical BER');
hold off ;


%4.d
%(i)

file = fopen('rice_odd.txt');
file_text = fscanf(file,'%c');
binascii = dec2bin(file_text);

 
%(ii)
bit_text_dec = bin2dec(binascii);

N_text=8; %bits
L_text = 2^N_text ; %number of quantized levels
D_text = (max(bit_text_dec) - min(bit_text_dec))/L_text; %step size
I_text = round((bit_text_dec - min(bit_text_dec))/D_text) ; %index
txt_quantized = min(bit_text_dec) + I_text*D_text;
figure(33);
n_txt = 1:1:485;
stem (n_txt,txt_quantized,'rx-');
title ('Quantized ASCII text');
xlabel('Numbered ascii characters');
ylabel('Decimal representation of ASCII code'); 

   
%(iii)
quantized_bits = dec2bin(round(txt_quantized));
quantized_bit_stream = zeros(3396, 1);
index_bit = 1;

for i = 1:485
    for j = 1:7
        if quantized_bits(i, j) == '1'
            quantized_bit_stream(index_bit) = 1;
        end
        index_bit = index_bit + 1;
    end
end


QPSK_txt_mod = qpskmod(quantized_bit_stream);
scatterplot(QPSK_txt_mod);
title('Constellation of QPSK modulated quantized text signal');

%(iv)

QPSK_txt_mod_noisy5dB = awgn(QPSK_txt_mod, 5);
QPSK_txt_mod_noisy15dB = awgn(QPSK_txt_mod, 15);
scatterplot(QPSK_txt_mod_noisy5dB);
title('QPSK constellation for Es/No = 5dB');
scatterplot(QPSK_txt_mod_noisy15dB);
title('QPSK constellation for Es/No = 15dB');

%(v)
qpskdemod_txt = comm.QPSKDemodulator('BitOutput',true); 
QPSK_txt_demod_5dB_dec = qpskdemod_txt(QPSK_txt_mod_noisy5dB);
QPSK_txt_demod_15dB_dec = qpskdemod_txt(QPSK_txt_mod_noisy15dB);


%(vi)

[num_qpsk_demod_5db, BER_qpsk_demod_5db] = symerr(quantized_bit_stream, QPSK_txt_demod_5dB_dec);
[num_qpsk_demod_15db, BER_qpsk_demod_15db] = symerr(quantized_bit_stream, QPSK_txt_demod_15dB_dec);
BER_qpsk_demod_5db_theoretical = qfunc(sqrt(10^0.5));
BER_qpsk_demod_15db_theoretical = qfunc(sqrt(10^1.5));


%(vii)

demod_file1  = char(bin2dec(reshape(char('0' + QPSK_txt_demod_15dB_dec(1:3395) ),7,[]).'))';
demod_file2  = char(bin2dec(reshape(char('0' + QPSK_txt_demod_5dB_dec(1:3395) ),7,[]).'))';
demod_text_5dB = fopen('demod_text_5db.txt', 'w');
fprintf(demod_text_5dB, '%c', demod_file2);
demod_text_15dB = fopen('demod_text_15db.txt', 'w');
fprintf(demod_text_15dB, '%c', demod_file1 );
    
    
    
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%% EXERCISE 5 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%5.a
[p,f_sound] = audioread('soundfile1_lab2.wav'); 
dt_p = 1/f_sound;
t_p = 0:dt_p:(length(p)*dt_p)-dt_p; %time(in seconds)
figure(41);
plot(t_p,p);
title ('Audio file');
xlabel ('Time[s]');
ylabel ('Amplitude');
 
%5.b
sound_bits =8; %bits
L_sound = 2^sound_bits ; %number of quantized levels
D_sound = (max(p) - min(p))/L_sound; %step size
I_sound = round((p - min(p))/D_sound) ; %index
x_sound = min(p) + I_sound*D_sound;
figure(42);
stem (t_p,x_sound,'rx-');
xlabel('Time[s]');
ylabel('Amplitude');
title('Quantized signal at 8 bits');

%5.c
A_s = x_sound*1000;

quantized_sound_bits = dec2bin(A_s,16) - '0';

quantized_sound_bit_stream = reshape(quantized_sound_bits.',[],1);
qpskmod_sound = comm.QPSKModulator('BitInput',true);
QPSK_sound_mod = qpskmod_sound(quantized_sound_bit_stream);
grid on;
scatterplot(QPSK_sound_mod);
title ('Constellation of QPSK modulated audio signal');


%5.d

QPSK_sound_mod_noisy4dB = awgn(QPSK_sound_mod, 4);
QPSK_sound_mod_noisy14dB = awgn(QPSK_sound_mod, 14);

scatterplot(QPSK_sound_mod_noisy4dB);
title( 'QPSK constellation for Es/No = 4dB');
scatterplot(QPSK_sound_mod_noisy14dB);
title( 'QPSK constellation for Es/No = 14dB');

%5.e
qpskdemod_sound = comm.QPSKDemodulator('BitOutput',true); 
QPSK_sound_demod_4dB_dec = qpskdemod_sound(QPSK_sound_mod_noisy4dB);
QPSK_sound_demod_14dB_dec = qpskdemod_sound(QPSK_sound_mod_noisy14dB);

%5.f
[num_qpsk_demod_4db, BER_qpsk_demod_4db] = symerr(quantized_sound_bit_stream, QPSK_sound_demod_4dB_dec);
[num_qpsk_demod_14db, BER_qpsk_demod_14db] = symerr(quantized_sound_bit_stream, QPSK_sound_demod_14dB_dec);
BER_qpsk_demod_4db_theoretical = qfunc(sqrt(10^0.4)) ;
BER_qpsk_demod_14db_theoretical = qfunc(sqrt(10^1.4)) ;

%5.g

new_sound1 = char('0'+ (reshape(QPSK_sound_demod_14dB_dec, 16, []).'));
new_sound_14dB= typecast(uint16(bin2dec(new_sound1)),'int16');
new_file_1 = double(abs(new_sound_14dB))./1000 ;
new_sound2 = char('0'+ (reshape(QPSK_sound_demod_4dB_dec, 16, []).'));
new_sound_4dB= typecast(uint16(bin2dec(new_sound2)),'int16');
new_file_2 = double(abs(new_sound_4dB))./1000 ;
new_file_1_norm = new_file_1/ max(new_file_1);
new_file_2_norm = new_file_2/ max(new_file_2);
audiowrite("demod_sound_14dB.wav", new_file_1_norm , 44100);
audiowrite("demod_sound_4dB.wav", new_file_2_norm, 44100);

%}