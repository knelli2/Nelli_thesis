# Use pdflatex
$pdf_mode = 1;
$bibtex_use = 2;

# Handle bibunits: run bibtex on each bu*.aux file
add_cus_dep('aux', 'bbl', 0, 'run_bibtex_on_bibunit');
sub run_bibtex_on_bibunit {
    my $base = $_[0];
    if ($base =~ /bu\d+$/) {
        return system("bibtex \"$base\"");
    }
    return 0;
}