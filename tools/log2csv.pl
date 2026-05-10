#!/usr/bin/perl
use strict;
use warnings;
use Text::CSV;

my $input_file = shift // 'train.log';   # 默认日志文件名
my $output_file = shift // 'training.csv'; # 默认输出 CSV 文件名

open(my $in, '<', $input_file) or die "Cannot open $input_file: $!";

# 创建一个 CSV 对象
my $csv = Text::CSV->new({ binary => 1, eol => "\n" });
open(my $out, '>', $output_file) or die "Cannot create $output_file: $!";

# 写入 CSV 头部
$csv->print($out, ['Epoch', 'D_Loss', 'G_Loss', 'Val_L1', 'Val_PSNR', 'LR_G', 'Saved_Best']);
my $current_epoch = undef;
my $d_loss = undef;
my $g_loss = undef;
my $val_l1 = undef;
my $val_psnr = undef;
my $lr_g = undef;
my $saved_best = 0;

while (my $line = <$in>) {
    chomp $line;

    # 跳过空行和配置头（可自行决定是否输出配置）
    next if $line =~ /^=+$/ || $line =~ /^Training Configuration:/ || $line =~ /^  / && $line !~ /Epoch/;

    # 匹配 epoch 摘要行
    if ($line =~ /Epoch \[(\d+)\/\d+\] \| D Loss: ([\d.]+) \| G Loss: ([\d.]+) \| Val L1: ([\d.]+) \| Val PSNR: ([\d.]+) dB \| LR_G: ([\de.+-]+)/) {
        # 如果上一个 epoch 已收集，先写入
        if (defined $current_epoch) {
            $csv->print($out, [$current_epoch, $d_loss, $g_loss, $val_l1, $val_psnr, $lr_g, $saved_best]);
        }
        $current_epoch = $1;
        $d_loss = $2;
        $g_loss = $3;
        $val_l1 = $4;
        $val_psnr = $5;
        $lr_g = $6;
        $saved_best = 0;  # 默认为否，如果下面有 "Saved best" 则设为 1
    }
    # 匹配 "Saved best" 行
    elsif ($line =~ /-> Saved best generator/) {
        $saved_best = 1;
    }
    # 匹配 "Checkpoint saved" 行（仅用于跳过，不影响当前 epoch）
    elsif ($line =~ /Checkpoint saved/) {
        # 不做特殊处理
    }
}

# 写入最后一个 epoch
if (defined $current_epoch) {
    $csv->print($out, [$current_epoch, $d_loss, $g_loss, $val_l1, $val_psnr, $lr_g, $saved_best]);
}

close $in;
close $out;

print "CSV file written to $output_file\n";
