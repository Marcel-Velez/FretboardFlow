import numpy as np
import torch
import math
from collections import defaultdict
import itertools


class GuitarMetrics:
    def __init__(self, n_strings=6, max_fret=25):
        self.n_strings = n_strings
        self.max_fret = max_fret

    ### 1. PITCH-BASED METRICS (PP, RP, F1P)
    def get_pitch_classes(self, diagram):
        """
        Converts a chord diagram (list of 6 frets) to a set of pitch classes.
        """
        tuning = [40, 45, 50, 55, 59, 64]  # Standard tuning: EADGBE
        pitch_classes = set()

        try:
            for string, fret in enumerate(diagram[:6]):
                if (fret-1) >= 0:  # Ignore muted strings (-1)
                    note = (tuning[string] + fret-1) % 12  # Convert to pitch class (0-11)
                    pitch_classes.add(note)
        except Exception as e:
            raise ValueError(f"failing for {len(diagram)=} error {e}\n {diagram=}")
        return pitch_classes


    def compute_pitch_metrics(self, predicted, ground_truth):
        """
        Computes Pitch Precision (PP), Pitch Recall (RP), and Pitch F1-score (F1P).
        """
        pred_pitches = self.get_pitch_classes(predicted)
        gt_pitches = self.get_pitch_classes(ground_truth)

        if len(pred_pitches) == 0 or len(gt_pitches) == 0:
            return 0, 0, 0  # Avoid division by zero

        PP = self.pc_precision(pred_pitches, gt_pitches)  # Precision
        RP = self.pc_recall(pred_pitches, gt_pitches)  # Recall

        out = 2*torch.div(PP*RP, PP+RP)
        F1P = torch.nan_to_num(out, nan=0, posinf=0, neginf=0)

        return PP, RP, F1P

    def pc_precision(self, pred_notes: set, expected_notes: set) -> float:
        size = len(pred_notes)
        if size == 0:
            return 1
        unwanted_notes = pred_notes.difference(expected_notes)
        return (size - len(unwanted_notes)) / size

    def pc_recall(self, pred_notes: set, expected_notes: set) -> float:
        size = len(expected_notes)
        if size == 0:
            return 1
        missing_notes = expected_notes.difference(pred_notes)
        return (size - len(missing_notes)) / size


    def binarize(self, tensor, with_mute=True) -> torch.Tensor:
        torch.roll(tensor, -1, dims=-1)
        # if with_mute:
        # max_tensor = torch.max(torch.unflatten(tensor, 1,
        #     (-1, 25)), dim=-1).values
        # pred_bin = torch.where(torch.unflatten(tensor, 1,
        #     (-1, 25)) == max_tensor[:, :, None], 1, 0)
        # # else:
        pred_bin = tensor > 0.5
        pred_bin = torch.unflatten(pred_bin, 1, (-1, 26))
        return pred_bin


    ### 2. TABLATURE-BASED METRICS (PSF, RSF, F1SF)
    def compute_tablature_metrics(self, predicted, ground_truth):
        """
        Computes String/Fret Precision (PSF), Recall (RSF), and F1-score (F1SF).
        """
        pred = self.binarize(predicted).view(6,26).float()
        PSF = self.stringfret_precision(pred, ground_truth)
        RSF = self.stringfret_recall(pred, ground_truth)
        F1SF = self.stringfret_f1(pred, ground_truth)

        return PSF, RSF, F1SF

    def stringfret_precision(self, fingering_vector: torch.Tensor, expected_fingering: torch.Tensor,
                             with_mute: bool = False, tol=1e-8) -> torch.Tensor:
        assert len(fingering_vector.shape) == 2, f"fingering vector not one hot encoded, shape: {fingering_vector.shape}"
        assert len(expected_fingering.shape) == 2, f"expected fingering not one hot encoded, shape: {expected_fingering.shape}"
        if with_mute:
            fingering_vector = fingering_vector[:, :, :-1]
            expected_fingering = expected_fingering[:, :, :-1]
        TP = torch.count_nonzero((fingering_vector + expected_fingering) == 2, dim=(-1, -2))
        FP = torch.count_nonzero((expected_fingering - fingering_vector) == -1, dim=(-1, -2))
        out = torch.div(TP, TP + FP)
        out = torch.nan_to_num(out, nan=0, posinf=0, neginf=0)
        return out

    def stringfret_recall(self, fingering_vector: torch.Tensor, expected_fingering: torch.Tensor,
                          with_mute: bool = False) -> torch.Tensor:
        if with_mute:
            fingering_vector = fingering_vector[:, :, :-1]
            expected_fingering = expected_fingering[:, :, :-1]
        TP = torch.count_nonzero((fingering_vector + expected_fingering) == 2, dim=(-1, -2))
        FN = torch.count_nonzero((fingering_vector - expected_fingering) == -1, dim=(-1, -2))
        out = torch.div(TP, TP + FN)
        out = torch.nan_to_num(out, nan=0, posinf=0, neginf=0)
        return out

    def stringfret_f1(self, fingering_vector, expected_fingering, with_mute: bool = False, tol=1e-8):
        if with_mute:
            fingering_vector = fingering_vector[:, :, :-1]
            expected_fingering = expected_fingering[:, :, :-1]
        TP = torch.count_nonzero((fingering_vector + expected_fingering) == 2, dim=(-1, -2))
        FP = torch.count_nonzero((expected_fingering - fingering_vector) == -1, dim=(-1, -2))
        FN = torch.count_nonzero((fingering_vector - expected_fingering) == -1, dim=(-1, -2))
        out = torch.div(2 * TP, 2 * TP + FP + FN)
        out = torch.nan_to_num(out, nan=0, posinf=0, neginf=0)
        return out




    def transition_cost(self, diagram1, diagram2, theta1=1.0, theta2=1.0, open_string_index=1):
        """
        Computes a simplified chord-to-chord transition cost.

        - diagram1, diagram2: Each is a length-n_strings array (or list)
          where:
            fret < 0 => muted,
            fret = 0 => open,
            fret > 0 => a fretted note.

        - theta1, theta2: Scaling factors for index-finger vs. total-finger movement.

        Returns a float in [0..1], where 1 means "very easy transition",
        and values closer to 0 mean "more difficult".
        """

        # Helper: get the "index-finger fret" = the minimum *positive* fret.
        # If a chord only has open/muted strings, we treat the index fret as 0.
        def get_index_fret(diag):
            fretted = [f for f in diag if f > open_string_index]  # strictly > 0
            return min(fretted) if fretted else 0

        # 1) "Wrist movement" = difference in index-fret positions.
        i1 = get_index_fret(diagram1)
        i2 = get_index_fret(diagram2)
        wrist_move = abs(i1 - i2)

        # 2) "Finger movement" = sum of absolute fret differences
        #    for each non-muted string in both chords.
        finger_move = 0
        for f1, f2 in zip(diagram1, diagram2):
            if (f1 >= 0) and (f2 >= 0):  # both played (open or fretted)
                finger_move += abs(f1 - f2)

        # 3) Combine into a single cost, normalized to (0..1].
        cost = 1 / (1 + theta1 * wrist_move + theta2 * finger_move)
        return cost

    def ratio_muted_strings(self, diagram, muted_string_id=0):
        """
        Returns fraction of strings that are muted.
        diagram: length-n_strings array
        """
        n_strings = len(diagram)
        num_muted = sum(1 for f in diagram if f == muted_string_id)
        return num_muted / n_strings

    def ratio_open_strings(self, diagram, open_string_id=1):
        """
        Returns fraction of *played* strings that are open
        (fret == 0). "Played" = not muted (fret >= 0).
        """
        num_open = sum(1 for f in diagram if f == open_string_id)
        strings_played = sum(1 for f in diagram if f >= open_string_id)

        if strings_played == 0:
            return 0.0
        return num_open / strings_played

    def string_centroid(self, diagram, muted_string_id=0):
        """
        A 'center of mass' index over the played strings.
        If no strings are played, returns 0.
        """

        n_strings = len(diagram)
        sum_str_idx = 0
        played_count = 0

        # We'll treat i=0 as "top" string and i = n_strings-1 as "bottom" string
        # and assign "string number" = n_strings - i to match the original code's logic.
        for i, fret in enumerate(diagram):
            if fret > muted_string_id:  # played (open or fretted)
                sum_str_idx += i+1 #(n_strings - i)
                played_count += 1

        if played_count == 0:
            return 0.0

        return sum_str_idx / played_count

    def ratio_unique_notes_pc(self, diagram, muted_string_id=0):
        pcs = self.get_pitch_classes(diagram)  # returns a set of ints in [0..11]
        played_count = sum(1 for f in diagram if f > muted_string_id)
        if played_count == 0:
            return 0.0
        return len(pcs) / played_count









    ##### anatomical score functions

    MIN_RANGE = [
        [0, 5, 15, 25],
        [5, 0, 6, 12],
        [15, 6, 0, 8.5],
        [25, 12, 8.5, 0]
    ]
    MAX_RANGE = [
        [0, 80, 95, 110],
        [80, 0, 52, 69],
        [95, 52, 0, 47],
        [110, 69, 47, 0]
    ]

    def anatomical_score(self, diagram):
        """
        Compute the "anatomical score" of a chord diagram, returning:

           (best_score, best_fingering_dict)

        where:
          - diagram is a length-n_strings array/list of int frets
               (<0 => muted, ==0 => open, >0 => fretted).
          - best_score is a float in [0..1], higher = more comfortable.
          - best_fingering_dict is the dictionary of {finger: [(string, fret), ...]}
            that yields the best score. It can be None if chord is only open strings.
        """
        # 1) Get set of all (string#, fret) pairs for strings that are NOT muted.
        sf_pairs = self._get_string_fret_set_numeric(diagram)

        # 2) If *all* these pairs are open (fret=0), we can short-circuit => score=1
        only_open = all(fret == 0 for (_, fret) in sf_pairs)
        if only_open and len(sf_pairs) > 0:
            return (1.0, None)  # no “fingering” needed if everything is open
        elif len(sf_pairs) == 0:
            # i.e. chord is fully muted or empty => no real “anatomical” challenge
            return (1.0, None)

        # 3) Exclude open strings from the assignment step, because in the
        #    original code we only assign fingers to *fretted* notes.
        fretted_pairs = [(s, f) for (s, f) in sf_pairs if f != 0]
        if len(fretted_pairs) == 0:
            # Everything was open or muted => also not challenging
            return (1.0, None)

        # 4) Generate all possible fingerings
        #    (which tries to assign each non-open note to some finger 1..4).
        fingerings = self._get_possible_fingerings(fretted_pairs)

        # 5) Score each fingering with _anatomical_score_dict and pick the best
        best_score = 0.0
        best_fingering = None
        for fingering_dict in fingerings:
            sc = self._anatomical_score_dict(fingering_dict)
            if sc > best_score:
                best_score = sc
                best_fingering = fingering_dict

        return (best_score, best_fingering)

    # -------------------------------------------------------------------------
    #  HELPER FUNCTIONS
    # -------------------------------------------------------------------------

    def _get_string_fret_set_numeric(self, diagram):
        """
        Convert an integer-based chord diagram to a list of (stringNum, fret).

        For consistency with the original code:
          - Start top string as "string_count = self.n_strings"
            and decrement down.
          - Skip muted (f<0).
          - Keep (string#, fret) for open (0) or fretted (>0).
        """
        out = []
        string_count = self.n_strings  # e.g. 6
        for fret in diagram:
            if fret < 0:
                # muted => skip entirely
                string_count -= 1
                continue
            else:
                out.append((string_count, fret))
                string_count -= 1
        return out

    def _get_possible_fingerings(self, sf_set):
        """
        Takes a list of (string#, fret) pairs (fretted only)
        and returns a list of dictionaries of the form:
            { fingerNumber : [ (string#, fret), ...],  ... }
        representing each unique way of assigning notes to fingers 1..4.

        The logic tries to ensure that if multiple notes are assigned to
        the same finger, they share the same fret. (i.e. a barre).
        """
        # Sort the pairs first by fret (ascending) and then by string# (descending).
        # This is the same ordering as in the original code.
        sorted_set = sorted(
            sorted(sf_set, key=lambda x: x[0], reverse=True),
            key=lambda x: x[1]
        )

        # If we have more than 4 notes, the code attempts combinations.
        # But the original approach basically picks finger combos
        # by enumerating (1,2,3,4) in all ways.
        num_notes = len(sorted_set)
        if num_notes >= 4:
            # We'll fill e.g. [1,1,1,...,2,3,4] in some combination
            # so that we definitely use 2,3,4 but maybe replicate finger 1 for extras
            # In practice, the original code tries "combinations" with some hack.
            # We'll replicate that logic:
            base = [1] * (num_notes - 3) + [2, 3, 4]  # if num_notes > 4
            # Generate permutations of these (some might be duplicates if num_notes=5 or so).
            # Actually the original used itertools.combinations to pick the assignment
            # of length = num_notes. We'll do a simpler approach that’s close enough:
            finger_assignments = set(itertools.permutations(base, num_notes))
        else:
            # If we have <=4 notes, we can just do permutations of (1..4) of length = #notes
            # or we can do combinations? The original used combinations, but permutations
            # might be even more thorough. We'll just do permutations for completeness.
            base = [1, 2, 3, 4][:num_notes]
            finger_assignments = set(itertools.permutations(base, num_notes))

        out = []
        for assignment in finger_assignments:
            # Build the dictionary {finger -> list of (string#, fret)}
            dico = defaultdict(list)
            for i, finger_id in enumerate(assignment):
                dico[finger_id].append(sorted_set[i])

            # Check the "barre" constraint: if multiple notes share the same finger,
            # they must be on the same fret. If not => skip.
            invalid = False
            for finger_id, pairs in dico.items():
                if len(pairs) > 1:
                    # They all must share the same fret
                    fret0 = pairs[0][1]
                    for (_, f) in pairs:
                        if f != fret0:
                            invalid = True
                            break
                if invalid:
                    break

            if not invalid:
                out.append(dico)

        return out

    def _anatomical_score_dict(self, fingering):
        """
        Given a single dictionary-based fingering:
            { fingerNumber : [ (string#, fret), ...], ... }
        compute a "comfort" score from 0..1+.
        The original code does a pairwise distance among all finger placements.
        """
        # If there's only 1 finger used and just 1 note, that's trivially easy
        if len(fingering) == 1:
            first_finger = list(fingering.keys())[0]
            if len(fingering[first_finger]) == 1:
                return 1.0

        # We'll do a double sum over pairs of fingers, pairs of placements
        # Then average by dividing by (m^2 - m), where m=#(fingers).
        fingers = list(fingering.keys())
        m = len(fingers)
        denom = max(m * m - m, 1)

        total_score = 0.0
        for i in range(m):
            for j in range(i + 1, m):
                f1, f2 = fingers[i], fingers[j]
                # Compare every note assigned to f1 with every note assigned to f2
                for (s1, fret1) in fingering[f1]:
                    for (s2, fret2) in fingering[f2]:
                        dist_mm = self._finger_distance(s1, fret1, s2, fret2)
                        # finger numbers are 1..4 => index them as 0..3
                        part_score = self._finger_score(dist_mm, f1 - 1, f2 - 1)
                        total_score += part_score

        return total_score / denom

    def _finger_distance(self, s1, f1, s2, f2,
                         gamma=36, scale_length=620,
                         nut_size=44.4, bridge_size=58.7):
        """
        Euclidean distance between two finger placements on the fretboard
        (string#, fret).  This wraps the same logic from the snippet:
            finger_distance = sqrt( fretDistance^2 + stringDistance^2 ).
        """
        # For the fret distance, we approximate positions with an exponential scale.
        f_dist = self._fret_distance(f1, f2, scale_length)

        # For the string distance, we factor how far apart the strings are
        # near whichever fret is closer.
        s_dist = self._string_distance(s1, s2, min(f1, f2),
                                       bridge_size=bridge_size,
                                       nut_size=nut_size,
                                       scale_length=scale_length,
                                       num_strings=self.n_strings)
        return math.sqrt(f_dist ** 2 + s_dist ** 2)

    def _fret_distance(self, m, n, scale_length):
        """
        Exponential approximation for distance (in mm) along the fretboard
        between fret m and fret n.
        """
        if m > n:
            m, n = n, m
        return scale_length * (2 ** (n / 12) - 2 ** (m / 12)) / (2 ** ((m + n) / 12))

    def _string_distance(self, s1, s2, fret, bridge_size, nut_size,
                         scale_length, num_strings):
        """
        Approximates the lateral (vertical) distance between two strings s1, s2
        at a given fret position.  The distance between strings narrows
        as you go up the neck.
        """
        # Convert s1, s2 in [1..6] to a fraction of total spacing
        string_hop = abs(s1 - s2) / (num_strings - 1)

        # The local "width" of the neck at fret is linearly interpolated
        # from nut_size to bridge_size:
        y_position = fret  # approximate: "fret" mm from the nut for interpolation
        dist_fret = (nut_size - bridge_size) * (y_position / scale_length) + bridge_size
        return dist_fret * string_hop

    def _finger_score(self, distance_mm, finger1, finger2, pinky_penalty=0.85):
        """
        The score for how comfortable it is for finger1 and finger2
        to be placed `distance_mm` apart.  Uses a piecewise polynomial
        fit from the snippet, referencing MIN_RANGE and MAX_RANGE.

        finger1, finger2 in {0..3} => index for [index, middle, ring, pinky].
        """
        a = self.MIN_RANGE[finger1][finger2]
        b = self.MAX_RANGE[finger1][finger2]
        # If either is "pinky" => multiply result by pinky_penalty
        if finger1 == 3 or finger2 == 3:
            penalty = pinky_penalty
        else:
            penalty = 1.0

        if distance_mm < a:
            # (1 + (x - 0.99*a)^3) * penalty
            return (1 + (distance_mm - 0.99 * a) ** 3) * penalty
        else:
            # (1 - ((x-0.99*a)/(1.01*b - 0.99*a))^2) * penalty
            denom = (1.01 * b - 0.99 * a)
            if denom <= 0:
                return 0.0
            frac = (distance_mm - 0.99 * a) / denom
            return (1 - frac ** 2) * penalty
