| suite     |   n_pairs |   n_examples | condition                   |   accuracy |   empty_rate |   pair_accuracy |   identical_answer_rate |   reference_nll |   boot_lo |   boot_hi |   boot_n |
|:----------|----------:|-------------:|:----------------------------|-----------:|-------------:|----------------:|------------------------:|----------------:|----------:|----------:|---------:|
| natural   |        96 |          192 | correct                     |     0.5333 |       0.0000 |          0.2708 |                  0.3438 |          1.3427 |  nan      |  nan      | nan      |
| natural   |        96 |          192 | complementary               |     0.2568 |       0.0000 |        nan      |                  0.3438 |          2.8072 |  nan      |  nan      | nan      |
| natural   |        96 |          192 | mismatched                  |     0.3125 |       0.0000 |        nan      |                  0.3438 |        nan      |  nan      |  nan      | nan      |
| natural   |        96 |          192 | no_image                    |     0.0000 |       0.0000 |        nan      |                  0.3438 |         21.4715 |  nan      |  nan      | nan      |
| natural   |        96 |          192 | correct_minus_complementary |     0.2766 |     nan      |        nan      |                nan      |          1.4645 |    0.1693 |    0.3797 |  96.0000 |
| grounding |        64 |          128 | correct                     |     0.4531 |       0.0000 |          0.2344 |                  0.4375 |          2.8891 |  nan      |  nan      | nan      |
| grounding |        64 |          128 | complementary               |     0.2891 |       0.0000 |        nan      |                  0.4375 |          3.7098 |  nan      |  nan      | nan      |
| grounding |        64 |          128 | mismatched                  |     0.3672 |       0.0000 |        nan      |                  0.4375 |        nan      |  nan      |  nan      | nan      |
| grounding |        64 |          128 | no_image                    |     0.0000 |       0.0000 |        nan      |                  0.4375 |         26.1323 |  nan      |  nan      | nan      |
| grounding |        64 |          128 | correct_minus_complementary |     0.1641 |     nan      |        nan      |                nan      |          0.8207 |    0.0312 |    0.3047 |  64.0000 |
