import tkinter as tk
import tkinter.font as tkfont
from pathlib import Path
import warnings
from tkinter import colorchooser, filedialog, messagebox, ttk
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from emicart.analysis.fft import (
    apply_frequency_domain_window_by_rbw,
    compute_single_sided_fft_db,
    get_window_array,
)
from emicart.analysis.units import SCOPE_BASE_UNITS, convert_trace_db
from emicart.instruments.tektronix import connect_to_scope, get_scope_data
from emicart.limits import registry as curve
from emicart.probes import registry as probe_registry
from emicart.ui.files import (
    get_default_save_dir,
    get_usb_save_dir,
    next_available_stem,
    normalize_stem,
)
from emicart.ui.import_export import (
    build_binary_payload,
    read_csv_import,
    read_mat_import,
    read_npz_import,
    write_csv_export,
)


def main():
    scope = None
    captured_volts = np.array([])
    sample_rate_captured = None
    traces = []
    next_trace_id = 1
    listbox_items = []
    default_save_dir = get_default_save_dir()

    root = tk.Tk()
    root.title("NASA Glenn EMI Precompliance Cart")
    root.geometry("1000x650")
    root.minsize(760, 480)
    icon_dir = Path(__file__).resolve().parents[1] / "data"
    icon_ico = icon_dir / "nasa_meatball.ico"
    icon_png = icon_dir / "nasa_meatball.png"
    try:
        if icon_ico.exists():
            root.iconbitmap(default=str(icon_ico))
        elif icon_png.exists():
            icon_image = tk.PhotoImage(file=str(icon_png))
            root.iconphoto(True, icon_image)
            root._icon_image = icon_image
    except Exception:
        pass
    try:
        root.state("zoomed")
    except Exception:
        pass

    colors = {
        "app_bg": "#F4F7FB",
        "card_bg": "#FFFFFF",
        "card_border": "#D8E2F0",
        "text_primary": "#132238",
        "text_muted": "#4B5D79",
        "accent": "#1F6FEB",
        "accent_hover": "#185BCC",
        "plot_bg": "#FBFDFF",
        "plot_grid_major": "#C9D7EA",
        "plot_grid_minor": "#E2EBF7",
        "plot_spine": "#AFC2DD",
        "plot_limit": "#D97706",
        "plot_measured": "#1D4ED8",
        "entry_bg": "#FAFCFF",
        "entry_border": "#B8C7DE",
        "status_bg": "#EAF1FC",
    }
    limit_style = {"color": colors["plot_limit"]}
    trace_palette = [
        "#1D4ED8",
        "#BE185D",
        "#0E7490",
        "#7C3AED",
        "#B45309",
        "#047857",
        "#C026D3",
        "#334155",
    ]

    root.configure(bg=colors["app_bg"])

    ui_font = tkfont.Font(family="Segoe UI", size=13)
    section_font = tkfont.Font(family="Segoe UI Semibold", size=14)

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass

    style.configure("App.TFrame", background=colors["app_bg"])
    style.configure(
        "Card.TFrame",
        background=colors["card_bg"],
        borderwidth=1,
        relief="solid",
        bordercolor=colors["card_border"],
    )
    style.configure("CardInner.TFrame", background=colors["card_bg"])
    style.configure(
        "SectionCard.TFrame",
        background=colors["card_bg"],
        borderwidth=1,
        relief="solid",
        bordercolor=colors["card_border"],
    )
    style.configure(
        "Ui.TLabel",
        font=ui_font,
        foreground=colors["text_primary"],
        background=colors["card_bg"],
    )
    style.configure(
        "Section.TLabel",
        font=section_font,
        foreground=colors["accent"],
        background=colors["card_bg"],
    )
    style.configure(
        "Hint.TLabel",
        font=ui_font,
        foreground=colors["text_muted"],
        background=colors["card_bg"],
    )
    style.configure(
        "Ui.TMenubutton",
        font=ui_font,
        padding=(10, 6),
        foreground=colors["text_primary"],
        background=colors["entry_bg"],
        bordercolor=colors["entry_border"],
        lightcolor=colors["entry_border"],
        darkcolor=colors["entry_border"],
    )
    style.map("Ui.TMenubutton", background=[("active", "#EEF4FF"), ("pressed", "#DDEBFF")])

    style.configure(
        "Primary.TButton",
        font=ui_font,
        padding=(12, 7),
        foreground="#FFFFFF",
        background=colors["accent"],
        bordercolor=colors["accent"],
        lightcolor=colors["accent"],
        darkcolor=colors["accent"],
    )
    style.map(
        "Primary.TButton",
        background=[("active", colors["accent_hover"]), ("pressed", colors["accent_hover"])],
    )

    style.configure(
        "Secondary.TButton",
        font=ui_font,
        padding=(12, 7),
        foreground=colors["text_primary"],
        background="#E8F0FE",
        bordercolor="#C4D6F7",
        lightcolor="#C4D6F7",
        darkcolor="#C4D6F7",
    )
    style.map("Secondary.TButton", background=[("active", "#DBE8FD"), ("pressed", "#D3E2FC")])

    style.configure(
        "Status.TLabel",
        font=ui_font,
        foreground=colors["text_primary"],
        background=colors["status_bg"],
    )
    style.configure(
        "Ui.TRadiobutton",
        font=ui_font,
        foreground=colors["text_primary"],
        background=colors["card_bg"],
    )

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    content_frame = ttk.Frame(root, style="App.TFrame", padding=(16, 12, 16, 12))
    content_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
    content_frame.columnconfigure(0, weight=7)
    content_frame.columnconfigure(1, weight=3)
    content_frame.rowconfigure(0, weight=1)

    plot_frame = ttk.Frame(content_frame, style="Card.TFrame", padding=(10, 10, 10, 10))
    plot_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
    plot_frame.columnconfigure(0, weight=1)
    plot_frame.rowconfigure(0, weight=1)

    fig = plt.figure(figsize=(5, 4), dpi=100, facecolor=colors["card_bg"])
    plt.subplot(111)

    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    control_card = ttk.Frame(content_frame, style="Card.TFrame")
    control_card.grid(row=0, column=1, sticky="nsew")
    control_card.columnconfigure(0, weight=1)
    control_card.rowconfigure(0, weight=1)

    control_frame = ttk.Frame(control_card, style="CardInner.TFrame", padding=(12, 12, 12, 12))
    control_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 6))

    for i in range(8):
        control_frame.rowconfigure(i, pad=8)
    # Spare vertical space should not stretch a single section.
    control_frame.rowconfigure(7, weight=1)
    control_frame.columnconfigure(0, weight=1)

    source_frame = ttk.Frame(control_frame, style="SectionCard.TFrame", padding=(10, 10, 10, 10))
    source_frame.grid(row=0, column=0, sticky="ew")
    source_frame.columnconfigure(0, weight=1)
    for i in range(13):
        source_frame.rowconfigure(i, pad=4)

    traces_frame = ttk.Frame(control_frame, style="SectionCard.TFrame", padding=(10, 10, 10, 10))
    traces_frame.grid(row=2, column=0, sticky="ew")
    traces_frame.columnconfigure(0, weight=1)
    traces_frame.rowconfigure(1, weight=0)

    actions_frame = ttk.Frame(control_frame, style="SectionCard.TFrame", padding=(10, 10, 10, 10))
    actions_frame.grid(row=1, column=0, sticky="ew")
    actions_frame.columnconfigure(0, weight=1)

    export_frame = ttk.Frame(control_frame, style="SectionCard.TFrame", padding=(10, 10, 10, 10))
    export_frame.grid(row=3, column=0, sticky="ew")
    export_frame.columnconfigure(0, weight=1)

    def add_section_header(parent, text):
        header = ttk.Frame(parent, style="CardInner.TFrame")
        header.grid(column=0, sticky="ew")
        header.columnconfigure(1, weight=1)
        accent_bar = tk.Frame(header, bg=colors["accent"], width=6, height=18)
        accent_bar.grid(row=0, column=0, sticky="ns", padx=(0, 8))
        ttk.Label(header, text=text, style="Section.TLabel").grid(row=0, column=1, sticky="w")
        return header

    standard_var = tk.StringVar(value="No Standard")
    previous_standard = standard_var.get()

    add_section_header(source_frame, "Source")
    ttk.Label(source_frame, text="Standard", style="Ui.TLabel").grid(row=1, column=0, sticky="w")
    standard_menu = ttk.OptionMenu(source_frame, standard_var, standard_var.get(), "No Standard")
    standard_menu.grid(row=2, column=0, sticky="ew")
    standard_menu.configure(style="Ui.TMenubutton")
    standard_menu["menu"].configure(font=ui_font)

    probe_names = probe_registry.get_probe_names()
    probe_var = tk.StringVar(value=probe_registry.get_default_probe().name)
    ttk.Label(source_frame, text="Probe", style="Ui.TLabel").grid(row=3, column=0, sticky="w")
    probe_menu = ttk.OptionMenu(source_frame, probe_var, probe_var.get(), *probe_names)
    probe_menu.grid(row=4, column=0, sticky="ew")
    probe_menu.configure(style="Ui.TMenubutton")
    probe_menu["menu"].configure(font=ui_font)
    manage_probes_button = ttk.Button(source_frame, text="Manage Probes", style="Secondary.TButton")
    manage_probes_button.grid(row=5, column=0, sticky="ew", pady=(4, 0))

    ttk.Label(source_frame, text="Limit Curve", style="Ui.TLabel").grid(row=6, column=0, sticky="w")
    curve_var = tk.StringVar(value="No Limit Curve")
    curve_menu = ttk.OptionMenu(source_frame, curve_var, curve_var.get(), "No Limit Curve")
    curve_menu.grid(row=7, column=0, sticky="ew")
    curve_menu.configure(style="Ui.TMenubutton")
    curve_menu["menu"].configure(font=ui_font)
    manage_standards_button = ttk.Button(source_frame, text="Manage Standards", style="Secondary.TButton")
    manage_standards_button.grid(row=8, column=0, sticky="ew", pady=(4, 0))

    window_choices = {
        "Raw FFT (No RBW Correction)": "raw",
        "No Windowing (Rectangular)": "none",
        "Hann": "hann",
        "Hamming": "hamming",
        "Blackman": "blackman",
        "Bartlett": "bartlett",
        "Flat Top": "flattop",
    }
    window_var = tk.StringVar(value="Raw FFT (No RBW Correction)")
    ttk.Label(source_frame, text="Window", style="Ui.TLabel").grid(row=9, column=0, sticky="w")
    window_menu = ttk.OptionMenu(source_frame, window_var, window_var.get(), *window_choices.keys())
    window_menu.grid(row=10, column=0, sticky="ew")
    window_menu.configure(style="Ui.TMenubutton")
    window_menu["menu"].configure(font=ui_font)
    window_shapes_button = ttk.Button(source_frame, text="Window Shapes", style="Secondary.TButton")
    window_shapes_button.grid(row=11, column=0, sticky="ew", pady=(4, 0))

    add_section_header(traces_frame, "Traces")
    trace_list_frame = ttk.Frame(traces_frame, style="CardInner.TFrame")
    trace_list_frame.grid(row=1, column=0, sticky="nsew")
    trace_list_frame.columnconfigure(0, weight=1)
    trace_list_frame.rowconfigure(0, weight=1)

    trace_listbox = tk.Listbox(
        trace_list_frame,
        height=4,
        font=ui_font,
        bg=colors["entry_bg"],
        fg=colors["text_primary"],
        selectbackground=colors["accent"],
        selectforeground="#FFFFFF",
        highlightthickness=1,
        highlightbackground=colors["entry_border"],
        bd=0,
        activestyle="none",
    )
    trace_listbox.grid(row=0, column=0, sticky="ew")
    trace_scroll = ttk.Scrollbar(trace_list_frame, orient="vertical", command=trace_listbox.yview)
    trace_scroll.grid(row=0, column=1, sticky="ns")
    trace_listbox.configure(yscrollcommand=trace_scroll.set)
    trace_actions_frame = ttk.Frame(traces_frame, style="CardInner.TFrame")
    trace_actions_frame.grid(row=2, column=0, sticky="ew", pady=(4, 0))
    trace_actions_frame.columnconfigure(0, weight=1)
    trace_actions_frame.columnconfigure(1, weight=1)
    trace_actions_frame.columnconfigure(2, weight=1)

    def refresh_standard_menu(preferred_standard=None):
        names = ["No Standard"]
        for std in curve.get_standards():
            if std not in names:
                names.append(std)
        menu = standard_menu["menu"]
        menu.delete(0, "end")
        for name in names:
            menu.add_command(label=name, command=tk._setit(standard_var, name))
        menu.configure(font=ui_font)

        current = preferred_standard or standard_var.get()
        if current in names:
            standard_var.set(current)
        else:
            standard_var.set(names[0])

    def refresh_curve_menu(*_):
        nonlocal previous_standard
        current_standard = standard_var.get()
        standard_changed = current_standard != previous_standard
        current_curve = curve_var.get()
        trace_units = {t.get("probe_units") for t in traces if t.get("probe_units")}
        locked_units = None
        mixed_trace_units = False
        if len(trace_units) == 1:
            locked_units = next(iter(trace_units))
        elif len(trace_units) > 1:
            mixed_trace_units = True
        if current_standard == "No Standard":
            names = ["No Limit Curve"]
            curve_menu.configure(state="disabled")
        else:
            curves_for_standard = curve.get_curves_for_standard(current_standard)
            if mixed_trace_units:
                curves_for_standard = []
            elif locked_units is not None:
                curves_for_standard = [c for c in curves_for_standard if c.units == locked_units]
            names = ["No Limit Curve"] + [c.name for c in curves_for_standard]
            curve_menu.configure(state="normal")

        menu = curve_menu["menu"]
        menu.delete(0, "end")
        for name in names:
            menu.add_command(label=name, command=tk._setit(curve_var, name))
        menu.configure(font=ui_font)

        if names:
            if current_curve in names:
                # When coming from "No Standard", auto-pick the first real limit curve.
                if (
                    current_standard != "No Standard"
                    and previous_standard == "No Standard"
                    and current_curve == "No Limit Curve"
                    and len(names) > 1
                ):
                    next_curve = names[1]
                else:
                    next_curve = current_curve
            elif current_standard != "No Standard" and len(names) > 1:
                next_curve = names[1]
            else:
                next_curve = names[0]
            curve_var.set(next_curve)

        # Standard changes can alter the active limit context even when curve_var text
        # does not change; force a list refresh to keep the limit entry in sync.
        refresh_trace_listbox()
        refresh_probe_menu()
        if standard_changed:
            selected_curve_name = curve_var.get()
            if current_standard == "No Standard":
                status_var.set("Standard set to No Standard (limit line disabled).")
            elif mixed_trace_units:
                status_var.set(
                    "Traces contain mixed units; limit-curve selection is restricted to No Limit Curve."
                )
            elif locked_units is not None and selected_curve_name == "No Limit Curve":
                status_var.set(
                    f"Standard set to {current_standard}; no compatible {locked_units} limit curve selected."
                )
            elif selected_curve_name == "No Limit Curve":
                status_var.set(f"Standard set to {current_standard}; no limit curve selected.")
            else:
                status_var.set(f"Selected limit curve: {selected_curve_name} ({current_standard})")
        previous_standard = current_standard

    def refresh_probe_menu(*_, preferred_name=None):
        names = probe_registry.get_probe_names()
        if not names:
            names = [probe_registry.get_default_probe().name]

        selected_standard = standard_var.get()
        selected_curve_name = curve_var.get()
        filtered_names = names
        if selected_standard != "No Standard" and selected_curve_name != "No Limit Curve":
            selected_curve = curve.get_curve_by_name(selected_curve_name, standard=selected_standard)
            if selected_curve is not None:
                target_units = selected_curve.units
                matching_names = []
                for name in names:
                    selected_probe = probe_registry.get_probe_by_name(name)
                    if selected_probe is None:
                        continue
                    # Strict filter: only probes whose measured units match the selected curve units.
                    if selected_probe.measured_units == target_units:
                        matching_names.append(name)
                if matching_names:
                    filtered_names = matching_names
                else:
                    filtered_names = []

        current_name = preferred_name or probe_var.get()
        menu = probe_menu["menu"]
        menu.delete(0, "end")
        if filtered_names:
            probe_menu.configure(state="normal")
            for name in filtered_names:
                menu.add_command(label=name, command=tk._setit(probe_var, name))
        else:
            probe_menu.configure(state="disabled")
            menu.add_command(label="No compatible probes", command=lambda: None)
        menu.configure(font=ui_font)

        if current_name in filtered_names:
            probe_var.set(current_name)
        elif filtered_names:
            probe_var.set(filtered_names[0])
        else:
            fallback = probe_registry.get_default_probe().name
            probe_var.set(fallback)

    def open_manage_probes_dialog():
        dialog = tk.Toplevel(root)
        dialog.title("Manage Probes")
        dialog.transient(root)
        dialog.grab_set()
        dialog.configure(bg=colors["card_bg"])
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        try:
            dialog.state("zoomed")
        except Exception:
            dialog.geometry(f"{screen_w}x{screen_h}+0+0")
        dialog.minsize(680, 500)
        dialog.columnconfigure(0, weight=1)
        dialog.rowconfigure(0, weight=1)

        wrapper = ttk.Frame(dialog, style="CardInner.TFrame", padding=(10, 10, 10, 10))
        wrapper.grid(row=0, column=0, sticky="nsew")
        wrapper.columnconfigure(0, weight=1)
        wrapper.rowconfigure(0, weight=1)
        panes = ttk.Panedwindow(wrapper, orient="horizontal")
        panes.grid(row=0, column=0, sticky="nsew")

        list_frame = ttk.Frame(panes, style="CardInner.TFrame")
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(1, weight=1)
        ttk.Label(list_frame, text="Probe List", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        probes_listbox = tk.Listbox(
            list_frame,
            font=ui_font,
            bg=colors["entry_bg"],
            fg=colors["text_primary"],
            selectbackground=colors["accent"],
            selectforeground="#FFFFFF",
            highlightthickness=1,
            highlightbackground=colors["entry_border"],
            bd=0,
            activestyle="none",
        )
        probes_listbox.grid(row=1, column=0, sticky="nsew")
        probes_scroll = ttk.Scrollbar(list_frame, orient="vertical", command=probes_listbox.yview)
        probes_scroll.grid(row=1, column=1, sticky="ns")
        probes_listbox.configure(yscrollcommand=probes_scroll.set)

        edit_frame = ttk.Frame(panes, style="CardInner.TFrame")
        edit_frame.columnconfigure(0, weight=1)
        for i in range(12):
            edit_frame.rowconfigure(i, pad=4)
        panes.add(list_frame, weight=2)
        panes.add(edit_frame, weight=3)

        ttk.Label(edit_frame, text="Name", style="Ui.TLabel").grid(row=0, column=0, sticky="w")
        name_var = tk.StringVar()
        name_entry = tk.Entry(
            edit_frame,
            textvariable=name_var,
            font=ui_font,
            bg=colors["entry_bg"],
            fg=colors["text_primary"],
            insertbackground=colors["text_primary"],
            relief="solid",
            highlightthickness=1,
            highlightbackground=colors["entry_border"],
            highlightcolor=colors["accent"],
            bd=0,
        )
        name_entry.grid(row=1, column=0, sticky="ew")

        ttk.Label(edit_frame, text="Units", style="Ui.TLabel").grid(row=2, column=0, sticky="w")
        units_var = tk.StringVar(value="dBuV")
        units_menu = ttk.OptionMenu(edit_frame, units_var, units_var.get(), "dBuV", "dBuA", "V/m")
        units_menu.grid(row=3, column=0, sticky="ew")
        units_menu.configure(style="Ui.TMenubutton")
        units_menu["menu"].configure(font=ui_font)

        ttk.Label(edit_frame, text="Impedance (ohms)", style="Ui.TLabel").grid(row=4, column=0, sticky="w")
        impedance_var = tk.StringVar()
        impedance_entry = tk.Entry(
            edit_frame,
            textvariable=impedance_var,
            font=ui_font,
            bg=colors["entry_bg"],
            fg=colors["text_primary"],
            insertbackground=colors["text_primary"],
            relief="solid",
            highlightthickness=1,
            highlightbackground=colors["entry_border"],
            highlightcolor=colors["accent"],
            bd=0,
        )
        impedance_entry.grid(row=5, column=0, sticky="ew")

        ttk.Label(edit_frame, text="E-Field Gain (V/m per V)", style="Ui.TLabel").grid(
            row=6, column=0, sticky="w"
        )
        gain_var = tk.StringVar()
        gain_entry = tk.Entry(
            edit_frame,
            textvariable=gain_var,
            font=ui_font,
            bg=colors["entry_bg"],
            fg=colors["text_primary"],
            insertbackground=colors["text_primary"],
            relief="solid",
            highlightthickness=1,
            highlightbackground=colors["entry_border"],
            highlightcolor=colors["accent"],
            bd=0,
        )
        gain_entry.grid(row=7, column=0, sticky="ew")

        ttk.Label(edit_frame, text="Description", style="Ui.TLabel").grid(row=8, column=0, sticky="w")
        description_var = tk.StringVar()
        description_entry = tk.Entry(
            edit_frame,
            textvariable=description_var,
            font=ui_font,
            bg=colors["entry_bg"],
            fg=colors["text_primary"],
            insertbackground=colors["text_primary"],
            relief="solid",
            highlightthickness=1,
            highlightbackground=colors["entry_border"],
            highlightcolor=colors["accent"],
            bd=0,
        )
        description_entry.grid(row=9, column=0, sticky="ew")

        hint_var = tk.StringVar(value="Select any probe to edit, duplicate, or delete.")
        ttk.Label(edit_frame, textvariable=hint_var, style="Hint.TLabel", justify="left").grid(
            row=10, column=0, sticky="w"
        )

        button_frame = ttk.Frame(edit_frame, style="CardInner.TFrame")
        button_frame.grid(row=11, column=0, sticky="ew", pady=(6, 0))
        for i in range(5):
            button_frame.columnconfigure(i, weight=1)

        def selected_probe_name_in_dialog():
            selected = probes_listbox.curselection()
            if not selected:
                return None
            return probes_listbox.get(selected[0])

        def load_probe_into_form(name):
            selected = probe_registry.get_probe_by_name(name)
            if selected is None:
                return
            name_var.set(selected.name)
            units_var.set(selected.measured_units)
            if selected.impedance_ohms is None:
                impedance_var.set("")
            else:
                impedance_var.set(f"{selected.impedance_ohms:g}")
            if selected.volts_to_v_per_m_gain is None:
                gain_var.set("")
            else:
                gain_var.set(f"{selected.volts_to_v_per_m_gain:g}")
            description_var.set(selected.description or "")
            hint_var.set("Probe selected: save updates, duplicate, or delete.")

        def refresh_dialog_probe_list(preferred=None):
            names = probe_registry.get_probe_names()
            probes_listbox.delete(0, tk.END)
            for name in names:
                probes_listbox.insert(tk.END, name)
            if not names:
                return
            target = preferred if preferred in names else names[0]
            idx = names.index(target)
            probes_listbox.selection_clear(0, tk.END)
            probes_listbox.selection_set(idx)
            probes_listbox.activate(idx)
            load_probe_into_form(target)

        def clear_form():
            name_var.set("")
            units_var.set("dBuV")
            impedance_var.set("")
            gain_var.set("")
            description_var.set("")
            hint_var.set("Create a probe and click Save.")
            name_entry.focus_set()

        def make_unique_probe_name(base_name):
            names = set(probe_registry.get_probe_names())
            candidate = base_name
            idx = 2
            while candidate in names:
                candidate = f"{base_name} {idx}"
                idx += 1
            return candidate

        def duplicate_probe():
            source_name = selected_probe_name_in_dialog()
            if not source_name:
                messagebox.showinfo("No Probe Selected", "Select a probe to duplicate.", parent=dialog)
                return
            source = probe_registry.get_probe_by_name(source_name)
            if source is None:
                return
            new_name = make_unique_probe_name(f"{source.name} Copy")
            name_var.set(new_name)
            units_var.set(source.measured_units)
            if source.impedance_ohms is None:
                impedance_var.set("")
            else:
                impedance_var.set(f"{source.impedance_ohms:g}")
            if source.volts_to_v_per_m_gain is None:
                gain_var.set("")
            else:
                gain_var.set(f"{source.volts_to_v_per_m_gain:g}")
            description_var.set(source.description or "")
            hint_var.set("Review fields and click Save to create duplicated probe.")
            name_entry.focus_set()
            name_entry.select_range(0, tk.END)

        def save_probe():
            name = name_var.get().strip()
            units = units_var.get().strip()
            imp_text = impedance_var.get().strip()
            if imp_text == "":
                impedance = None
            else:
                try:
                    impedance = float(imp_text)
                except ValueError:
                    messagebox.showerror("Invalid Impedance", "Impedance must be numeric.", parent=dialog)
                    return
            gain_text = gain_var.get().strip()
            if gain_text == "":
                gain = None
            else:
                try:
                    gain = float(gain_text)
                except ValueError:
                    messagebox.showerror("Invalid Gain", "E-field gain must be numeric.", parent=dialog)
                    return
            try:
                probe_registry.upsert_probe(
                    name=name,
                    measured_units=units,
                    impedance_ohms=impedance,
                    volts_to_v_per_m_gain=gain,
                    description=description_var.get().strip(),
                )
            except ValueError as e:
                messagebox.showerror("Save Probe Failed", str(e), parent=dialog)
                return
            refresh_dialog_probe_list(preferred=name)
            refresh_probe_menu(preferred_name=name)
            status_var.set(f"Saved probe '{name}'.")

        def delete_probe():
            name = selected_probe_name_in_dialog()
            if not name:
                return
            if not messagebox.askyesno("Delete Probe", f"Delete probe '{name}'?", parent=dialog):
                return
            if probe_registry.delete_probe(name):
                fallback = probe_var.get()
                if fallback == name:
                    fallback = probe_registry.get_default_probe().name
                refresh_dialog_probe_list(preferred=fallback)
                refresh_probe_menu(preferred_name=fallback)
                status_var.set(f"Deleted probe '{name}'.")

        def on_list_select(*_):
            name = selected_probe_name_in_dialog()
            if name:
                load_probe_into_form(name)

        ttk.Button(button_frame, text="New", command=clear_form, style="Secondary.TButton").grid(
            row=0, column=0, sticky="ew", padx=(0, 4)
        )
        ttk.Button(button_frame, text="Duplicate", command=duplicate_probe, style="Secondary.TButton").grid(
            row=0, column=1, sticky="ew", padx=4
        )
        ttk.Button(button_frame, text="Save", command=save_probe, style="Primary.TButton").grid(
            row=0, column=2, sticky="ew", padx=4
        )
        ttk.Button(button_frame, text="Delete", command=delete_probe, style="Secondary.TButton").grid(
            row=0, column=3, sticky="ew", padx=4
        )
        ttk.Button(button_frame, text="Close", command=dialog.destroy, style="Secondary.TButton").grid(
            row=0, column=4, sticky="ew", padx=(4, 0)
        )

        probes_listbox.bind("<<ListboxSelect>>", on_list_select)
        refresh_dialog_probe_list(preferred=probe_var.get())

    def open_manage_standards_dialog():
        curve.reload_standards()
        dialog = tk.Toplevel(root)
        dialog.title("Manage Standards and Curves")
        dialog.transient(root)
        dialog.grab_set()
        dialog.configure(bg=colors["card_bg"])
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        try:
            dialog.state("zoomed")
        except Exception:
            dialog.geometry(f"{screen_w}x{screen_h}+0+0")
        dialog.minsize(900, 620)
        dialog.columnconfigure(0, weight=1)
        dialog.rowconfigure(0, weight=1)

        wrapper = ttk.Frame(dialog, style="CardInner.TFrame", padding=(10, 10, 10, 10))
        wrapper.grid(row=0, column=0, sticky="nsew")
        wrapper.columnconfigure(0, weight=1)
        wrapper.rowconfigure(0, weight=1)
        panes = ttk.Panedwindow(wrapper, orient="horizontal")
        panes.grid(row=0, column=0, sticky="nsew")

        standards_frame = ttk.Frame(panes, style="CardInner.TFrame")
        standards_frame.columnconfigure(0, weight=1)
        standards_frame.rowconfigure(1, weight=1)
        ttk.Label(standards_frame, text="Standards", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        standards_listbox = tk.Listbox(
            standards_frame,
            font=ui_font,
            bg=colors["entry_bg"],
            fg=colors["text_primary"],
            selectbackground=colors["accent"],
            selectforeground="#FFFFFF",
            highlightthickness=1,
            highlightbackground=colors["entry_border"],
            bd=0,
            activestyle="none",
            exportselection=False,
        )
        standards_listbox.grid(row=1, column=0, sticky="nsew")
        standards_scroll = ttk.Scrollbar(standards_frame, orient="vertical", command=standards_listbox.yview)
        standards_scroll.grid(row=1, column=1, sticky="ns")
        standards_listbox.configure(yscrollcommand=standards_scroll.set)

        curves_frame = ttk.Frame(panes, style="CardInner.TFrame")
        curves_frame.columnconfigure(0, weight=1)
        curves_frame.rowconfigure(1, weight=1)
        ttk.Label(curves_frame, text="Limit Curves", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        curves_listbox = tk.Listbox(
            curves_frame,
            font=ui_font,
            bg=colors["entry_bg"],
            fg=colors["text_primary"],
            selectbackground=colors["accent"],
            selectforeground="#FFFFFF",
            highlightthickness=1,
            highlightbackground=colors["entry_border"],
            bd=0,
            activestyle="none",
            exportselection=False,
        )
        curves_listbox.grid(row=1, column=0, sticky="nsew")
        curves_scroll = ttk.Scrollbar(curves_frame, orient="vertical", command=curves_listbox.yview)
        curves_scroll.grid(row=1, column=1, sticky="ns")
        curves_listbox.configure(yscrollcommand=curves_scroll.set)

        edit_frame = ttk.Frame(panes, style="CardInner.TFrame")
        edit_frame.columnconfigure(0, weight=1)
        edit_frame.rowconfigure(7, weight=1)
        edit_frame.rowconfigure(9, weight=1)
        panes.add(standards_frame, weight=2)
        panes.add(curves_frame, weight=2)
        panes.add(edit_frame, weight=4)

        standard_name_var = tk.StringVar()
        curve_name_var = tk.StringVar()
        units_var = tk.StringVar(value="dBuV")
        hint_var = tk.StringVar(value="Select any standard/curve to edit, duplicate, or delete.")

        ttk.Label(edit_frame, text="Standard Name", style="Ui.TLabel").grid(row=0, column=0, sticky="w")
        standard_entry = tk.Entry(
            edit_frame,
            textvariable=standard_name_var,
            font=ui_font,
            bg=colors["entry_bg"],
            fg=colors["text_primary"],
            insertbackground=colors["text_primary"],
            relief="solid",
            highlightthickness=1,
            highlightbackground=colors["entry_border"],
            highlightcolor=colors["accent"],
            bd=0,
        )
        standard_entry.grid(row=1, column=0, sticky="ew")

        ttk.Label(edit_frame, text="Curve Name", style="Ui.TLabel").grid(row=2, column=0, sticky="w")
        curve_entry = tk.Entry(
            edit_frame,
            textvariable=curve_name_var,
            font=ui_font,
            bg=colors["entry_bg"],
            fg=colors["text_primary"],
            insertbackground=colors["text_primary"],
            relief="solid",
            highlightthickness=1,
            highlightbackground=colors["entry_border"],
            highlightcolor=colors["accent"],
            bd=0,
        )
        curve_entry.grid(row=3, column=0, sticky="ew")

        ttk.Label(edit_frame, text="Units", style="Ui.TLabel").grid(row=4, column=0, sticky="w")
        units_menu = ttk.OptionMenu(edit_frame, units_var, units_var.get(), "dBuV", "dBuA", "V/m")
        units_menu.grid(row=5, column=0, sticky="ew")
        units_menu.configure(style="Ui.TMenubutton")
        units_menu["menu"].configure(font=ui_font)

        ttk.Label(
            edit_frame, text="Breakpoints (Hz,Level) one per line, comma separated", style="Ui.TLabel"
        ).grid(row=6, column=0, sticky="w")
        breakpoints_text = tk.Text(
            edit_frame,
            height=9,
            font=ui_font,
            bg=colors["entry_bg"],
            fg=colors["text_primary"],
            insertbackground=colors["text_primary"],
            relief="solid",
            highlightthickness=1,
            highlightbackground=colors["entry_border"],
            highlightcolor=colors["accent"],
            bd=0,
        )
        breakpoints_text.grid(row=7, column=0, sticky="nsew")

        ttk.Label(
            edit_frame, text="RBW per segment (Hz), one per line, blank = none", style="Ui.TLabel"
        ).grid(row=8, column=0, sticky="w")
        rbw_text = tk.Text(
            edit_frame,
            height=6,
            font=ui_font,
            bg=colors["entry_bg"],
            fg=colors["text_primary"],
            insertbackground=colors["text_primary"],
            relief="solid",
            highlightthickness=1,
            highlightbackground=colors["entry_border"],
            highlightcolor=colors["accent"],
            bd=0,
        )
        rbw_text.grid(row=9, column=0, sticky="nsew")

        ttk.Label(edit_frame, textvariable=hint_var, style="Hint.TLabel", justify="left").grid(
            row=10, column=0, sticky="w", pady=(6, 0)
        )

        button_frame = ttk.Frame(edit_frame, style="CardInner.TFrame")
        button_frame.grid(row=11, column=0, sticky="ew", pady=(8, 0))
        for i in range(5):
            button_frame.columnconfigure(i, weight=1)

        def selected_standard_name():
            s = standards_listbox.curselection()
            if not s:
                return None
            return standards_listbox.get(s[0])

        def selected_curve_name():
            s = curves_listbox.curselection()
            if not s:
                return None
            return curves_listbox.get(s[0])

        def parse_breakpoints_from_text() -> List[Tuple[float, float]]:
            lines = [ln.strip() for ln in breakpoints_text.get("1.0", tk.END).splitlines() if ln.strip()]
            points: List[Tuple[float, float]] = []
            for line in lines:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) != 2:
                    raise ValueError("Each breakpoint line must be 'frequency_hz, level_db'.")
                points.append((float(parts[0]), float(parts[1])))
            return points

        def parse_rbw_from_text(expected_segments: int) -> List[Optional[float]]:
            lines = [ln.strip() for ln in rbw_text.get("1.0", tk.END).splitlines()]
            if len(lines) == 1 and lines[0] == "":
                return [None] * expected_segments
            lines = [ln for ln in lines if ln != ""]
            if len(lines) != expected_segments:
                raise ValueError(
                    f"RBW must have exactly {expected_segments} line(s), one for each curve segment."
                )
            out: List[Optional[float]] = []
            for line in lines:
                if line == "":
                    out.append(None)
                else:
                    value = float(line)
                    out.append(value if value > 0 else None)
            return out

        def populate_curve_form(standard_name: str, curve_name: str):
            c = curve.get_curve_by_name(curve_name, standard=standard_name)
            if c is None:
                return
            standard_name_var.set(standard_name)
            curve_name_var.set(c.name)
            units_var.set(c.units)
            breakpoints_text.delete("1.0", tk.END)
            breakpoints_text.insert(
                "1.0",
                "\n".join(f"{x:.12g}, {y:.12g}" for x, y in c.breakpoints),
            )
            rbw_text.delete("1.0", tk.END)
            rbw_text.insert(
                "1.0",
                "\n".join("" if r is None else f"{float(r):.12g}" for r in c.resolution_bandwidth_hz),
            )
            hint_var.set("Curve selected: Save updates, duplicate, or delete.")

        def refresh_standard_list(preferred_standard=None):
            names = curve.get_standards()
            standards_listbox.delete(0, tk.END)
            for name in names:
                standards_listbox.insert(tk.END, name)
            if not names:
                return
            target = preferred_standard if preferred_standard in names else names[0]
            idx = names.index(target)
            standards_listbox.selection_clear(0, tk.END)
            standards_listbox.selection_set(idx)
            standards_listbox.activate(idx)
            standard_name_var.set(target)
            refresh_curve_list(target)

        def refresh_curve_list(standard_name: str, preferred_curve=None, auto_select_curve=True):
            curve_names = [c.name for c in curve.get_curves_for_standard(standard_name)]
            curves_listbox.delete(0, tk.END)
            for cname in curve_names:
                curves_listbox.insert(tk.END, cname)
            if not curve_names:
                return
            if not auto_select_curve:
                curves_listbox.selection_clear(0, tk.END)
                curves_listbox.activate(0)
                return
            target = preferred_curve if preferred_curve in curve_names else curve_names[0]
            idx = curve_names.index(target)
            curves_listbox.selection_clear(0, tk.END)
            curves_listbox.selection_set(idx)
            curves_listbox.activate(idx)
            populate_curve_form(standard_name, target)

        def clear_curve_form():
            curve_name_var.set("")
            units_var.set("dBuV")
            breakpoints_text.delete("1.0", tk.END)
            rbw_text.delete("1.0", tk.END)
            std = standard_name_var.get().strip()
            if std:
                hint_var.set("Standard selected: enter values and Save to create a new curve.")
            else:
                hint_var.set("Enter values and Save to create a curve.")
            curve_entry.focus_set()

        def make_unique_curve_name(standard_name: str, base_name: str) -> str:
            existing = {c.name for c in curve.get_curves_for_standard(standard_name)}
            candidate = base_name
            idx = 2
            while candidate in existing:
                candidate = f"{base_name} {idx}"
                idx += 1
            return candidate

        def make_unique_standard_name(base_name: str) -> str:
            existing = set(curve.get_standards())
            candidate = base_name
            idx = 2
            while candidate in existing:
                candidate = f"{base_name} {idx}"
                idx += 1
            return candidate

        def duplicate_standard(std: str):
            curves_for_std = curve.get_curves_for_standard(std)
            if not curves_for_std:
                messagebox.showinfo("No Curves", f"Standard '{std}' has no curves to duplicate.", parent=dialog)
                return
            new_std = make_unique_standard_name(f"{std} Copy")
            try:
                for c in curves_for_std:
                    curve.upsert_curve(
                        standard=new_std,
                        curve_name=c.name,
                        units=c.units,
                        breakpoints=list(c.breakpoints),
                        resolution_bandwidth_hz=list(c.resolution_bandwidth_hz),
                    )
            except Exception as e:
                messagebox.showerror("Duplicate Failed", str(e), parent=dialog)
                return
            refresh_standard_menu(preferred_standard=new_std)
            refresh_curve_menu()
            refresh_standard_list(preferred_standard=new_std)
            status_var.set(f"Duplicated standard '{std}' to '{new_std}'.")
            hint_var.set("Standard selected: duplicated standard and selected first curve.")

        def duplicate_curve():
            std = selected_standard_name()
            cname = selected_curve_name()
            if not std:
                messagebox.showinfo("No Standard Selected", "Select a standard to duplicate.", parent=dialog)
                return
            if not cname:
                duplicate_standard(std)
                return
            c = curve.get_curve_by_name(cname, standard=std)
            if c is None:
                return
            standard_name_var.set(std)
            curve_name_var.set(make_unique_curve_name(std, f"{c.name} Copy"))
            units_var.set(c.units)
            breakpoints_text.delete("1.0", tk.END)
            breakpoints_text.insert("1.0", "\n".join(f"{x:.12g}, {y:.12g}" for x, y in c.breakpoints))
            rbw_text.delete("1.0", tk.END)
            rbw_text.insert(
                "1.0",
                "\n".join("" if r is None else f"{float(r):.12g}" for r in c.resolution_bandwidth_hz),
            )
            hint_var.set("Review values and click Save to create a duplicated curve.")
            curve_entry.focus_set()
            curve_entry.select_range(0, tk.END)

        def save_curve():
            std = standard_name_var.get().strip()
            cname = curve_name_var.get().strip()
            units = units_var.get().strip()
            try:
                points = parse_breakpoints_from_text()
                rbw_values = parse_rbw_from_text(max(len(points) - 1, 0))
                curve.upsert_curve(
                    standard=std,
                    curve_name=cname,
                    units=units,
                    breakpoints=points,
                    resolution_bandwidth_hz=rbw_values,
                )
            except Exception as e:
                messagebox.showerror("Save Failed", str(e), parent=dialog)
                return

            refresh_standard_menu(preferred_standard=std)
            refresh_curve_menu()
            refresh_standard_list(preferred_standard=std)
            refresh_curve_list(std, preferred_curve=cname)
            status_var.set(f"Saved curve '{cname}' in standard '{std}'.")

        def delete_selected():
            std = selected_standard_name() or standard_name_var.get().strip()
            cname = selected_curve_name()
            if not std:
                messagebox.showinfo("Nothing Selected", "Select a standard or curve to delete.", parent=dialog)
                return
            if cname:
                if not messagebox.askyesno("Delete Curve", f"Delete curve '{cname}'?", parent=dialog):
                    return
                if curve.delete_curve(std, cname):
                    refresh_standard_menu(preferred_standard=std)
                    refresh_curve_menu()
                    refresh_standard_list(preferred_standard=std)
                    status_var.set(f"Deleted curve '{cname}'.")
                return
            if not messagebox.askyesno("Delete Standard", f"Delete standard '{std}' and all curves?", parent=dialog):
                return
            if curve.delete_standard(std):
                refresh_standard_menu(preferred_standard="No Standard")
                refresh_curve_menu()
                refresh_standard_list()
                clear_curve_form()
                status_var.set(f"Deleted standard '{std}'.")

        def on_standard_select(*_):
            std = selected_standard_name()
            if std:
                standard_name_var.set(std)
                refresh_curve_list(std, auto_select_curve=False)
                clear_curve_form()
                hint_var.set("Standard selected: enter values and Save to create a new curve.")

        def on_curve_select(*_):
            std = selected_standard_name()
            cname = selected_curve_name()
            if std and cname:
                populate_curve_form(std, cname)

        ttk.Button(button_frame, text="New", command=clear_curve_form, style="Secondary.TButton").grid(
            row=0, column=0, sticky="ew", padx=(0, 4)
        )
        ttk.Button(button_frame, text="Duplicate", command=duplicate_curve, style="Secondary.TButton").grid(
            row=0, column=1, sticky="ew", padx=4
        )
        ttk.Button(button_frame, text="Save", command=save_curve, style="Primary.TButton").grid(
            row=0, column=2, sticky="ew", padx=4
        )
        ttk.Button(button_frame, text="Delete Selected", command=delete_selected, style="Secondary.TButton").grid(
            row=0, column=3, sticky="ew", padx=4
        )
        ttk.Button(button_frame, text="Close", command=dialog.destroy, style="Secondary.TButton").grid(
            row=0, column=4, sticky="ew", padx=(4, 0)
        )

        standards_listbox.bind("<<ListboxSelect>>", on_standard_select)
        standards_listbox.bind("<ButtonRelease-1>", on_standard_select)
        standards_listbox.bind("<KeyRelease-Up>", on_standard_select)
        standards_listbox.bind("<KeyRelease-Down>", on_standard_select)
        curves_listbox.bind("<<ListboxSelect>>", on_curve_select)
        curves_listbox.bind("<ButtonRelease-1>", on_curve_select)
        curves_listbox.bind("<KeyRelease-Up>", on_curve_select)
        curves_listbox.bind("<KeyRelease-Down>", on_curve_select)
        preferred_std = standard_var.get()
        if preferred_std == "No Standard":
            preferred_std = None
        refresh_standard_list(preferred_standard=preferred_std)

    def open_window_shapes_dialog():
        dialog = tk.Toplevel(root)
        dialog.title("Window Shapes")
        dialog.transient(root)
        dialog.configure(bg=colors["card_bg"])
        dialog.geometry("900x560")
        dialog.minsize(760, 460)
        dialog.columnconfigure(0, weight=1)
        dialog.rowconfigure(0, weight=1)

        frame = ttk.Frame(dialog, style="CardInner.TFrame", padding=(10, 10, 10, 10))
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        fig_w = plt.figure(figsize=(8, 4), dpi=100, facecolor=colors["card_bg"])
        ax_w = fig_w.add_subplot(111)
        ax_w.set_facecolor(colors["plot_bg"])

        # Group aliases that map to the same underlying window kernel.
        kernel_to_labels = {}
        for label, key in window_choices.items():
            kernel = "rectangular" if key in {"none", "raw"} else key
            kernel_to_labels.setdefault(kernel, []).append(label)

        n = 512
        x = np.linspace(0.0, 1.0, n)
        shape_palette = [
            "#1D4ED8",
            "#BE185D",
            "#0E7490",
            "#7C3AED",
            "#B45309",
            "#047857",
            "#C026D3",
            "#334155",
        ]

        for idx, (kernel, labels) in enumerate(kernel_to_labels.items()):
            if kernel == "rectangular":
                w = np.ones(n, dtype=float)
            else:
                w = get_window_array(kernel, n)
            w = np.asarray(w, dtype=float)
            if np.max(np.abs(w)) > 0:
                w = w / np.max(np.abs(w))
            label_text = " / ".join(labels)
            ax_w.plot(x, w, linewidth=2.0, color=shape_palette[idx % len(shape_palette)], label=label_text)

        ax_w.set_xlabel("Normalized Sample Index", color=colors["text_primary"])
        ax_w.set_ylabel("Normalized Amplitude", color=colors["text_primary"])
        ax_w.set_title("Window Function Shapes", color=colors["text_primary"], pad=10)
        ax_w.tick_params(axis="both", colors=colors["text_muted"])
        ax_w.grid(True, which="major", ls="-", lw=0.7, color=colors["plot_grid_major"], alpha=0.8)
        for spine in ax_w.spines.values():
            spine.set_color(colors["plot_spine"])
            spine.set_linewidth(1.0)
        legend = ax_w.legend(frameon=True, loc="best")
        legend.get_frame().set_facecolor("#FFFFFF")
        legend.get_frame().set_edgecolor(colors["card_border"])
        legend.get_frame().set_alpha(0.95)

        canvas_w = FigureCanvasTkAgg(fig_w, master=frame)
        canvas_w.draw()
        canvas_w.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def get_selected_curve():
        selected_standard = standard_var.get()
        if selected_standard == "No Standard" or curve_var.get() == "No Limit Curve":
            return None, selected_standard
        available_curves = curve.get_curves_for_standard(selected_standard)
        if not available_curves:
            return None, selected_standard
        selected_curve = available_curves[0]
        selected_curve = curve.get_curve_by_name(curve_var.get(), standard=selected_standard) or selected_curve
        return selected_curve, selected_standard

    def get_selected_probe():
        return probe_registry.get_probe_by_name(probe_var.get()) or probe_registry.get_default_probe()

    def probe_to_snapshot(probe):
        return {
            "probe_name": probe.name,
            "probe_units": probe.measured_units,
            "probe_impedance_ohms": probe.impedance_ohms,
            "probe_v_per_m_gain": probe.volts_to_v_per_m_gain,
        }

    def get_trace_probe(trace):
        units = trace.get("probe_units")
        if not units:
            # Backward compatibility for older traces/csv entries.
            named = probe_registry.get_probe_by_name(trace.get("probe_name", ""))
            if named is None:
                named = probe_registry.get_probe_by_name(probe_var.get())
            if named is None:
                named = probe_registry.get_default_probe()
            # Freeze inferred probe onto the trace so later probe dropdown changes
            # do not retroactively rescale this trace.
            trace.update(probe_to_snapshot(named))
            return named
        return probe_registry.Probe(
            name=trace.get("probe_name", "Imported Probe"),
            measured_units=units,
            impedance_ohms=trace.get("probe_impedance_ohms"),
            volts_to_v_per_m_gain=trace.get("probe_v_per_m_gain"),
            description="",
        )

    def get_trace_label():
        nonlocal next_trace_id
        label = f"Capture {next_trace_id}"
        next_trace_id += 1
        return label

    def refresh_trace_listbox(*_, select_index=None):
        nonlocal listbox_items
        selected_curve, _ = get_selected_curve()
        listbox_items = []
        trace_listbox.delete(0, tk.END)

        if selected_curve is not None:
            listbox_items.append({"kind": "limit"})
            trace_listbox.insert(tk.END, f"L: {selected_curve.name}")

        for idx, trace in enumerate(traces):
            listbox_items.append({"kind": "trace", "index": idx})
            trace_listbox.insert(tk.END, f"{idx + 1}: {trace['label']}")

        if not listbox_items:
            return

        if select_index is None or select_index < 0 or select_index >= len(listbox_items):
            select_index = len(listbox_items) - 1
        trace_listbox.selection_clear(0, tk.END)
        trace_listbox.selection_set(select_index)
        trace_listbox.activate(select_index)

    def render_plot(selected_curve):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Attempt to set non-positive xlim on a log-scaled axis will be ignored.",
                category=UserWarning,
            )
            fig.clear()
        if selected_curve is not None:
            gs = fig.add_gridspec(2, 1, height_ratios=[5.0, 1.4], hspace=0.08)
            ax = fig.add_subplot(gs[0])
            ax_rbw = fig.add_subplot(gs[1], sharex=ax)
        else:
            ax = fig.add_subplot(111)
            ax_rbw = None

        ax.set_facecolor(colors["plot_bg"])
        plot_label_size = max(10, ui_font.cget("size") + 1)
        plot_tick_size = max(9, ui_font.cget("size"))
        plot_title_size = max(12, ui_font.cget("size") + 3)
        plot_legend_size = max(9, ui_font.cget("size"))
        selected_probe = get_selected_probe()
        if selected_curve is not None:
            plot_units = selected_curve.units
        elif len(traces) > 0:
            plot_units = traces[0].get("probe_units") or selected_probe.measured_units
        else:
            plot_units = selected_probe.measured_units

        if selected_curve is not None:
            x_min = selected_curve.breakpoints[0][0]
            x_max = selected_curve.breakpoints[-1][0]
            limit_freqs = np.logspace(np.log10(x_min), np.log10(x_max), 2000)
            limit_y = selected_curve.get_curve(limit_freqs)
            limit_y = np.array([np.nan if v is None else v for v in limit_y], dtype=float)
            ax.semilogx(
                limit_freqs,
                limit_y,
                label=selected_curve.name,
                color=limit_style["color"],
                linewidth=3.0,
                linestyle=(0, (8, 4)),
                zorder=5,
            )

        for idx, trace in enumerate(traces):
            trace_color = trace.get("color", trace_palette[idx % len(trace_palette)])
            trace_windowed = trace["windowed"]
            if plot_units != SCOPE_BASE_UNITS:
                try:
                    trace_probe = get_trace_probe(trace)
                    trace_windowed = convert_trace_db(
                        trace_windowed, SCOPE_BASE_UNITS, plot_units, trace_probe
                    )
                except ValueError:
                    trace_windowed = trace["windowed"]
            ax.semilogx(
                trace["freqs"],
                trace_windowed,
                label=trace["label"],
                color=trace_color,
                linewidth=1.8,
                alpha=0.82,
                zorder=3,
            )

        ax.set_xlabel("Frequency (Hz)", color=colors["text_primary"], fontsize=plot_label_size)
        ax.set_ylabel(f"Amplitude ({plot_units})", color=colors["text_primary"], fontsize=plot_label_size)
        title = (
            f"{selected_curve.name} vs Measured Spectrum"
            if selected_curve is not None
            else "Measured Spectrum"
        )
        ax.set_title(title, color=colors["text_primary"], pad=12, fontsize=plot_title_size)
        ax.tick_params(axis="both", colors=colors["text_muted"], labelsize=plot_tick_size)

        for spine in ax.spines.values():
            spine.set_color(colors["plot_spine"])
            spine.set_linewidth(1.0)

        ax.grid(True, which="major", ls="-", lw=0.8, color=colors["plot_grid_major"], alpha=0.75)
        ax.grid(True, which="minor", ls="-", lw=0.6, color=colors["plot_grid_minor"], alpha=0.8)

        handles, labels = ax.get_legend_handles_labels()
        if handles and labels:
            legend = ax.legend(frameon=True, loc="best", fontsize=plot_legend_size)
            legend.get_frame().set_facecolor("#FFFFFF")
            legend.get_frame().set_edgecolor(colors["card_border"])
            legend.get_frame().set_alpha(0.95)

        if ax_rbw is not None:
            # Guard log scaling against non-positive default/shared limits.
            x_lo_raw, x_hi_raw = ax.get_xlim()
            if x_lo_raw <= 0 or x_hi_raw <= 0:
                if selected_curve is not None and selected_curve.breakpoints:
                    x_min_curve = max(float(selected_curve.breakpoints[0][0]), 1e-3)
                    x_max_curve = max(float(selected_curve.breakpoints[-1][0]), x_min_curve * 10.0)
                    ax.set_xlim(x_min_curve, x_max_curve)
                else:
                    min_freq = None
                    max_freq = None
                    for trace in traces:
                        freqs = np.asarray(trace.get("freqs", []), dtype=float)
                        if freqs.size == 0:
                            continue
                        positive = freqs[freqs > 0]
                        if positive.size == 0:
                            continue
                        fmin = float(np.min(positive))
                        fmax = float(np.max(positive))
                        min_freq = fmin if min_freq is None else min(min_freq, fmin)
                        max_freq = fmax if max_freq is None else max(max_freq, fmax)
                    if min_freq is None or max_freq is None or max_freq <= min_freq:
                        ax.set_xlim(1.0, 10.0)
                    else:
                        ax.set_xlim(min_freq, max_freq)

            ax_rbw.set_facecolor(colors["plot_bg"])
            ax_rbw.set_xscale("log")
            ax_rbw.set_ylim(0, 1)
            ax_rbw.set_yticks([])
            ax_rbw.set_ylabel("RBW", color=colors["text_muted"], fontsize=plot_label_size)
            ax_rbw.tick_params(axis="x", colors=colors["text_muted"], labelsize=plot_tick_size)

            rbw_band_colors = ["#DCE8FA", "#E8F2FF"]
            xs = [bp[0] for bp in selected_curve.breakpoints]
            rbws = selected_curve.resolution_bandwidth_hz
            x_lo, x_hi = ax.get_xlim()
            if x_lo > x_hi:
                x_lo, x_hi = x_hi, x_lo

            rbw_segments = []
            if rbws:
                # Extrapolate below the first breakpoint using the first RBW.
                if x_lo < xs[0]:
                    rbw_segments.append((x_lo, min(x_hi, xs[0]), rbws[0]))

                # In-range RBW segments.
                for i in range(len(rbws)):
                    seg_start = max(x_lo, xs[i])
                    seg_end = min(x_hi, xs[i + 1])
                    if seg_end > seg_start:
                        rbw_segments.append((seg_start, seg_end, rbws[i]))

                # Extrapolate above the last breakpoint using the last RBW.
                if x_hi > xs[-1]:
                    rbw_segments.append((max(x_lo, xs[-1]), x_hi, rbws[-1]))

            for i, (x0, x1, rbw_val) in enumerate(rbw_segments):
                if x1 <= x0:
                    continue
                ax_rbw.axvspan(
                    x0,
                    x1,
                    ymin=0.1,
                    ymax=0.9,
                    facecolor=rbw_band_colors[i % len(rbw_band_colors)],
                    edgecolor=colors["card_border"],
                    linewidth=1.0,
                    alpha=0.95,
                )

                if rbw_val is None:
                    rbw_text = "N/A"
                elif rbw_val >= 1e6:
                    rbw_text = f"{rbw_val / 1e6:g} MHz"
                elif rbw_val >= 1e3:
                    rbw_text = f"{rbw_val / 1e3:g} kHz"
                else:
                    rbw_text = f"{rbw_val:g} Hz"

                x_mid = (x0 * x1) ** 0.5
                ax_rbw.text(
                    x_mid,
                    0.5,
                    rbw_text,
                    ha="center",
                    va="center",
                    color=colors["text_primary"],
                    fontsize=max(9, ui_font.cget("size")),
                )

            for spine in ax_rbw.spines.values():
                spine.set_color(colors["plot_spine"])
                spine.set_linewidth(1.0)

            plt.setp(ax.get_xticklabels(), visible=False)
            ax_rbw.grid(True, which="major", ls="-", lw=0.6, color=colors["plot_grid_minor"], alpha=0.7)

        if ax_rbw is not None:
            fig.subplots_adjust(left=0.09, right=0.98, top=0.94, bottom=0.12, hspace=0.08)
        else:
            fig.subplots_adjust(left=0.09, right=0.98, top=0.92, bottom=0.12)
        canvas.draw()

    def rerender_from_selection(*_):
        selected_curve, _ = get_selected_curve()
        render_plot(selected_curve)

    def on_probe_changed(*_):
        selected_curve, _ = get_selected_curve()
        selected_probe = get_selected_probe()
        if selected_curve is not None and selected_curve.units != SCOPE_BASE_UNITS:
            if selected_probe.can_convert(SCOPE_BASE_UNITS, selected_curve.units):
                if selected_curve.units == "dBuA" and selected_probe.impedance_ohms is not None:
                    status_var.set(
                        f"Probe '{selected_probe.name}' active ({selected_probe.impedance_ohms:g} ohm): "
                        f"plotting in {selected_curve.units}."
                    )
                elif selected_curve.units == "V/m" and selected_probe.volts_to_v_per_m_gain is not None:
                    status_var.set(
                        f"Probe '{selected_probe.name}' active ({selected_probe.volts_to_v_per_m_gain:g} V/m per V): "
                        f"plotting in {selected_curve.units}."
                    )
                else:
                    status_var.set(
                        f"Probe '{selected_probe.name}' active: plotting in {selected_curve.units}."
                    )
            else:
                status_var.set(
                    f"Probe '{selected_probe.name}' cannot convert {SCOPE_BASE_UNITS} to {selected_curve.units}."
                )
        else:
            status_var.set(f"Probe selected: {selected_probe.name}")
        render_plot(selected_curve)

    def on_window_changed(*_):
        selected_curve, selected_standard = get_selected_curve()
        if selected_curve is None and selected_standard != "No Standard":
            status_var.set(f"No curves defined for {selected_standard}.")
            render_plot(selected_curve)
            return

        if len(traces) == 0:
            render_plot(selected_curve)
            return

        try:
            selected_window = window_choices[window_var.get()]
            reapplied = 0
            skipped = 0

            def summarize_effective_rbw(target_rbw_hz):
                if target_rbw_hz is None:
                    return None
                rbw_arr = np.array(
                    [np.nan if v is None else float(v) for v in target_rbw_hz],
                    dtype=float,
                )
                valid = rbw_arr[np.isfinite(rbw_arr) & (rbw_arr > 0)]
                if valid.size == 0:
                    return None
                if np.allclose(valid, valid[0], rtol=0.0, atol=1e-12):
                    return float(valid[0])
                return None

            for trace in traces:
                if trace.get("sample_rate") is None or len(trace.get("volts", [])) == 0:
                    skipped += 1
                    continue

                trace["window"] = window_var.get()
                if selected_window == "raw":
                    windowed_base = np.array(trace["original"], copy=True)
                elif selected_curve is not None:
                    target_rbw_hz = selected_curve.get_resolution_bandwidth(trace["freqs"])
                    kernel_window = "boxcar" if selected_window == "none" else selected_window
                    windowed_base = apply_frequency_domain_window_by_rbw(
                        mags_db=trace["original"],
                        freqs_hz=trace["freqs"],
                        target_rbw_hz=target_rbw_hz,
                        window_name=kernel_window,
                    )
                    trace["effective_rbw_hz"] = summarize_effective_rbw(target_rbw_hz)
                else:
                    # Without a selected curve there is no RBW profile. Fall back to
                    # legacy whole-record time-domain windowing behavior.
                    kernel_window = "rectangular" if selected_window == "none" else selected_window
                    _, windowed_base = compute_single_sided_fft_db(
                        trace["volts"],
                        trace["sample_rate"],
                        window_name=kernel_window,
                    )
                    trace["effective_rbw_hz"] = None
                trace["windowed"] = windowed_base
                reapplied += 1

            render_plot(selected_curve)
            if skipped > 0:
                status_var.set(
                    f"Applied window={window_var.get()} to {reapplied} trace(s); skipped {skipped} imported trace(s)."
                )
            else:
                status_var.set(f"Applied window={window_var.get()} to {reapplied} trace(s).")
        except Exception as e:
            status_var.set(f"Window update failed: {e}")

    manage_probes_button.configure(command=open_manage_probes_dialog)
    manage_standards_button.configure(command=open_manage_standards_dialog)
    window_shapes_button.configure(command=open_window_shapes_dialog)

    def create_trace_from_capture(selected_curve, trace_label):
        selected_probe = get_selected_probe()
        selected_window = window_choices[window_var.get()]
        freqs_captured, spectrum_original_captured = compute_single_sided_fft_db(
            captured_volts,
            sample_rate_captured,
            window_name="rectangular",
        )
        effective_rbw_used_hz = None
        if selected_window == "raw":
            spectrum_windowed_captured = np.array(spectrum_original_captured, copy=True)
        elif selected_curve is not None:
            target_rbw_hz = selected_curve.get_resolution_bandwidth(freqs_captured)
            kernel_window = "boxcar" if selected_window == "none" else selected_window
            spectrum_windowed_captured = apply_frequency_domain_window_by_rbw(
                mags_db=spectrum_original_captured,
                freqs_hz=freqs_captured,
                target_rbw_hz=target_rbw_hz,
                window_name=kernel_window,
            )
            rbw_arr = np.array(
                [np.nan if v is None else float(v) for v in target_rbw_hz],
                dtype=float,
            )
            valid_rbw = rbw_arr[np.isfinite(rbw_arr) & (rbw_arr > 0)]
            if valid_rbw.size > 0 and np.allclose(valid_rbw, valid_rbw[0], rtol=0.0, atol=1e-12):
                effective_rbw_used_hz = float(valid_rbw[0])
        else:
            # Without a selected curve there is no RBW profile. Fall back to
            # legacy whole-record time-domain windowing behavior.
            kernel_window = "rectangular" if selected_window == "none" else selected_window
            _, spectrum_windowed_captured = compute_single_sided_fft_db(
                captured_volts,
                sample_rate_captured,
                window_name=kernel_window,
            )

        return {
            "label": trace_label,
            "window": window_var.get(),
            "effective_rbw_hz": effective_rbw_used_hz,
            "color": trace_palette[len(traces) % len(trace_palette)],
            "volts": np.array(captured_volts, copy=True),
            "sample_rate": float(sample_rate_captured),
            "freqs": freqs_captured,
            "original": spectrum_original_captured,
            "windowed": spectrum_windowed_captured,
            **probe_to_snapshot(selected_probe),
        }

    def append_trace_from_capture(selected_curve):
        trace_label = get_trace_label()
        trace = create_trace_from_capture(selected_curve, trace_label)
        traces.append(trace)
        refresh_curve_menu()
        selected_curve_now, _ = get_selected_curve()
        offset = 1 if selected_curve_now is not None else 0
        refresh_trace_listbox(select_index=offset + len(traces) - 1)
        render_plot(selected_curve)
        return trace

    def edit_selected_trace_popup(*_):
        selected_items = trace_listbox.curselection()
        if not selected_items:
            status_var.set("Select a trace or limit line to edit.")
            return
        selected_idx = selected_items[0]
        if selected_idx < 0 or selected_idx >= len(listbox_items):
            status_var.set("Selected trace index is out of range.")
            return
        item = listbox_items[selected_idx]
        if item["kind"] == "limit":
            selected_curve, selected_standard = get_selected_curve()
            if selected_curve is None:
                status_var.set("No active limit line to edit.")
                return

            dialog = tk.Toplevel(root)
            dialog.title("Edit Limit Line")
            dialog.transient(root)
            dialog.grab_set()
            dialog.resizable(False, False)
            dialog.configure(bg=colors["card_bg"])

            color_var = tk.StringVar(value=limit_style["color"])

            ttk.Label(dialog, text=f"Limit: {selected_curve.name}", style="Ui.TLabel").grid(
                row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(10, 4)
            )
            ttk.Label(dialog, text="Line Color", style="Ui.TLabel").grid(row=1, column=0, sticky="w", padx=10, pady=(6, 4))
            color_preview = tk.Label(
                dialog,
                text="      ",
                bg=color_var.get(),
                relief="solid",
                bd=1,
                font=ui_font,
            )
            color_preview.grid(row=1, column=1, sticky="e", padx=10, pady=(6, 4))

            def pick_limit_color():
                _, picked = colorchooser.askcolor(
                    initialcolor=color_var.get(), parent=dialog, title="Select Limit Line Color"
                )
                if picked:
                    color_var.set(picked)
                    color_preview.configure(bg=picked)

            ttk.Button(dialog, text="Choose Color", command=pick_limit_color, style="Secondary.TButton").grid(
                row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10)
            )

            button_frame = ttk.Frame(dialog, style="App.TFrame")
            button_frame.grid(row=3, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))
            button_frame.columnconfigure(0, weight=1)
            button_frame.columnconfigure(1, weight=1)

            def save_limit_changes():
                limit_style["color"] = color_var.get()
                render_plot(selected_curve)
                status_var.set(f"Updated limit line color for {selected_standard}.")
                dialog.destroy()

            def cancel_limit_changes():
                dialog.destroy()

            ttk.Button(button_frame, text="Save", command=save_limit_changes, style="Primary.TButton").grid(
                row=0, column=0, sticky="ew", padx=(0, 4)
            )
            ttk.Button(button_frame, text="Cancel", command=cancel_limit_changes, style="Secondary.TButton").grid(
                row=0, column=1, sticky="ew", padx=(4, 0)
            )
            return

        trace_idx = item["index"]
        if trace_idx < 0 or trace_idx >= len(traces):
            status_var.set("Selected trace index is out of range.")
            return

        trace = traces[trace_idx]
        dialog = tk.Toplevel(root)
        dialog.title(f"Edit Trace {trace_idx + 1}")
        dialog.transient(root)
        dialog.grab_set()
        dialog.resizable(False, False)
        dialog.configure(bg=colors["card_bg"])

        name_var = tk.StringVar(value=trace["label"])
        color_var = tk.StringVar(value=trace.get("color", trace_palette[trace_idx % len(trace_palette)]))
        trace_probe = get_trace_probe(trace)
        selected_curve_for_trace, _ = get_selected_curve()
        target_probe_units = (
            selected_curve_for_trace.units if selected_curve_for_trace is not None else trace_probe.measured_units
        )
        compatible_probe_names = []
        for probe_name in probe_registry.get_probe_names():
            candidate_probe = probe_registry.get_probe_by_name(probe_name)
            if candidate_probe is None:
                continue
            if candidate_probe.measured_units == target_probe_units:
                compatible_probe_names.append(probe_name)
        if trace_probe.name not in compatible_probe_names:
            compatible_probe_names.insert(0, trace_probe.name)
        if not compatible_probe_names:
            compatible_probe_names = [trace_probe.name]
        probe_name_var = tk.StringVar(
            value=trace_probe.name if trace_probe.name in compatible_probe_names else compatible_probe_names[0]
        )

        ttk.Label(dialog, text="Trace Label", style="Ui.TLabel").grid(row=0, column=0, sticky="w", padx=10, pady=(10, 4))
        name_entry = tk.Entry(
            dialog,
            textvariable=name_var,
            width=30,
            font=ui_font,
            bg=colors["entry_bg"],
            fg=colors["text_primary"],
            insertbackground=colors["text_primary"],
            relief="solid",
            highlightthickness=1,
            highlightbackground=colors["entry_border"],
            highlightcolor=colors["accent"],
            bd=0,
        )
        name_entry.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10)
        name_entry.focus_set()
        name_entry.select_range(0, tk.END)

        ttk.Label(dialog, text="Probe Used", style="Ui.TLabel").grid(row=2, column=0, sticky="w", padx=10, pady=(10, 4))
        probe_menu = ttk.OptionMenu(dialog, probe_name_var, probe_name_var.get(), *compatible_probe_names)
        probe_menu.grid(row=3, column=0, columnspan=2, sticky="ew", padx=10)
        probe_menu.configure(style="Ui.TMenubutton")
        probe_menu["menu"].configure(font=ui_font)

        ttk.Label(dialog, text="Trace Color", style="Ui.TLabel").grid(row=4, column=0, sticky="w", padx=10, pady=(10, 4))
        color_preview = tk.Label(
            dialog,
            text="      ",
            bg=color_var.get(),
            relief="solid",
            bd=1,
            font=ui_font,
        )
        color_preview.grid(row=4, column=1, sticky="e", padx=10, pady=(10, 4))

        def pick_color():
            _, picked = colorchooser.askcolor(initialcolor=color_var.get(), parent=dialog, title="Select Trace Color")
            if picked:
                color_var.set(picked)
                color_preview.configure(bg=picked)

        pick_button = ttk.Button(dialog, text="Choose Color", command=pick_color, style="Secondary.TButton")
        pick_button.grid(row=5, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))

        button_frame = ttk.Frame(dialog, style="App.TFrame")
        button_frame.grid(row=6, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        button_frame.columnconfigure(2, weight=1)

        def save_changes():
            new_label = name_var.get().strip()
            if not new_label:
                status_var.set("Capture label cannot be empty.")
                return
            trace["label"] = new_label
            trace["color"] = color_var.get()
            selected_probe_name = probe_name_var.get().strip()
            selected_probe = probe_registry.get_probe_by_name(selected_probe_name)
            if selected_probe is not None:
                trace.update(probe_to_snapshot(selected_probe))
            else:
                trace.update(probe_to_snapshot(trace_probe))
            selected_curve_now, _ = get_selected_curve()
            offset = 1 if selected_curve_now is not None else 0
            refresh_trace_listbox(select_index=offset + trace_idx)
            selected_curve, selected_standard = get_selected_curve()
            render_plot(selected_curve)
            if selected_curve is not None:
                status_var.set(f"Updated trace {trace_idx + 1} for {selected_standard}.")
            else:
                status_var.set(f"Updated trace {trace_idx + 1}.")
            dialog.destroy()

        def delete_trace():
            if not messagebox.askyesno(
                "Delete Trace",
                f"Delete trace '{trace['label']}'?",
                parent=dialog,
            ):
                return
            deleted_label = trace["label"]
            del traces[trace_idx]
            refresh_curve_menu()
            selected_curve, selected_standard = get_selected_curve()
            if selected_curve is not None:
                render_plot(selected_curve)
            selected_curve_now, _ = get_selected_curve()
            offset = 1 if selected_curve_now is not None else 0
            refresh_trace_listbox(select_index=min(offset + trace_idx, offset + len(traces) - 1))
            status_var.set(f"Deleted trace '{deleted_label}'.")
            dialog.destroy()

        def cancel():
            dialog.destroy()

        dialog.bind("<Return>", lambda _e: save_changes())

        save_button = ttk.Button(button_frame, text="Save", command=save_changes, style="Primary.TButton")
        save_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        delete_button = ttk.Button(button_frame, text="Delete", command=delete_trace, style="Secondary.TButton")
        delete_button.grid(row=0, column=1, sticky="ew", padx=3)
        cancel_button = ttk.Button(button_frame, text="Cancel", command=cancel, style="Secondary.TButton")
        cancel_button.grid(row=0, column=2, sticky="ew", padx=(6, 0))

    def clear_selected_trace():
        if len(traces) == 0:
            status_var.set("No traces available.")
            return
        selected_items = trace_listbox.curselection()
        if not selected_items:
            status_var.set("Select a trace to clear.")
            return
        idx = selected_items[0]
        if idx < 0 or idx >= len(listbox_items):
            status_var.set("Selected trace index is out of range.")
            return
        item = listbox_items[idx]
        if item["kind"] != "trace":
            status_var.set("Limit line cannot be cleared.")
            return
        trace_idx = item["index"]
        removed = traces[trace_idx]["label"]
        del traces[trace_idx]
        refresh_curve_menu()
        selected_curve, _ = get_selected_curve()
        render_plot(selected_curve)
        selected_curve_now, _ = get_selected_curve()
        offset = 1 if selected_curve_now is not None else 0
        refresh_trace_listbox(select_index=min(offset + trace_idx, offset + len(traces) - 1))
        status_var.set(f"Cleared trace '{removed}'.")

    def clear_all_traces():
        if len(traces) == 0:
            status_var.set("No traces available.")
            return
        if not messagebox.askyesno("Clear All Traces", "Remove all traces from the plot?", parent=root):
            return
        traces.clear()
        refresh_curve_menu()
        selected_curve, _ = get_selected_curve()
        render_plot(selected_curve)
        refresh_trace_listbox()
        status_var.set("Cleared all traces.")

    def update_plot():
        status_var.set("Updating plot...")
        selected_curve, selected_standard = get_selected_curve()
        if selected_curve is None and selected_standard != "No Standard":
            status_var.set(f"No curves defined for {selected_standard}.")
            return

        nonlocal scope
        nonlocal captured_volts
        nonlocal sample_rate_captured

        try:
            if scope is None:
                status_var.set("Connecting to scope...")
                root.update_idletasks()
                scope = connect_to_scope()

            captured_volts, dt = get_scope_data(scope)
            sample_rate_captured = 1 / dt
            trace = append_trace_from_capture(selected_curve)
            if selected_curve is None:
                status_var.set(f"Added trace '{trace['label']}' ({len(captured_volts)} points, no limit curve).")
            else:
                status_var.set(
                    f"Added trace '{trace['label']}' ({len(captured_volts)} points, window={trace['window']})."
                )
        except Exception as e:
            status_var.set(f"Scope capture failed: {e}")

    def import_data():
        import_dir, _ = resolve_save_dir()
        if import_dir is None:
            import_dir = default_save_dir
        path = filedialog.askopenfilename(
            title="Import EmiCart Data",
            initialdir=str(import_dir),
            filetypes=[
                ("EmiCart data", "*.csv *.npz *.mat"),
                ("CSV files", "*.csv"),
                ("NumPy files", "*.npz"),
                ("MATLAB files", "*.mat"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        spinner_frames = ["|", "/", "-", "\\"]
        spinner_idx = 0

        def tick_import_status(stage: str):
            nonlocal spinner_idx
            frame = spinner_frames[spinner_idx % len(spinner_frames)]
            spinner_idx += 1
            status_var.set(f"{frame} Importing data: {stage}")
            root.update()

        tick_import_status("starting")
        path_obj = Path(path)
        suffix = path_obj.suffix.lower()

        clear_before_import = True
        if len(traces) > 0:
            clear_before_import = messagebox.askyesno(
                "Import Data",
                "Clear existing traces before import?\n\n"
                "Yes = clear first (recommended)\n"
                "No = append imported traces",
                parent=root,
            )

        legacy_window_labels = {
            "No Windowing": "No Windowing (Rectangular)",
            "Square (Rectangular)": "No Windowing (Rectangular)",
        }

        def finalize_import(imported_traces, meta, source_name: str):
            import_standard = meta.get("Standard")
            import_curve = meta.get("LimitCurve")
            import_probe = meta.get("Probe")
            import_window = meta.get("WindowSelection")
            if import_window in legacy_window_labels:
                import_window = legacy_window_labels[import_window]

            imported = len(imported_traces)
            for i, trace in enumerate(imported_traces, start=1):
                if i % 200 == 0:
                    tick_import_status(f"building {source_name} traces ({i}/{imported})")
                if not trace.get("color"):
                    trace["color"] = trace_palette[len(traces) % len(trace_palette)]
                traces.append(trace)
            if imported == 0:
                status_var.set("No valid trace data found in selected file.")
                return

            tick_import_status(f"finalizing {source_name}: refreshing menus")
            refresh_curve_menu()
            if import_probe:
                tick_import_status(f"finalizing {source_name}: restoring probe")
                refresh_probe_menu(preferred_name=import_probe)
            if import_standard:
                tick_import_status(f"finalizing {source_name}: restoring standard/curve")
                refresh_standard_menu(preferred_standard=import_standard)
                if standard_var.get() == "No Standard":
                    curve_var.set("No Limit Curve")
                elif import_curve and import_curve != "No Limit Curve":
                    available_curve_names = [c.name for c in curve.get_curves_for_standard(standard_var.get())]
                    if import_curve in available_curve_names:
                        curve_var.set(import_curve)
                elif import_curve == "No Limit Curve":
                    curve_var.set("No Limit Curve")
            if not import_window:
                inferred = imported_traces[0].get("window")
                if inferred in legacy_window_labels:
                    inferred = legacy_window_labels[inferred]
                if inferred in window_choices:
                    import_window = inferred
            if import_window in window_choices:
                tick_import_status(f"finalizing {source_name}: restoring window")
                window_var.set(import_window)

            tick_import_status(f"finalizing {source_name}: rendering plot")
            selected_curve, _ = get_selected_curve()
            render_plot(selected_curve)
            tick_import_status(f"finalizing {source_name}: updating trace list")
            refresh_trace_listbox(select_index=len(traces) - 1 if len(traces) > 0 else None)
            mode = "after clearing existing traces" if clear_before_import else "appended to existing traces"
            status_var.set(f"Imported {imported} trace(s) from {path} ({mode}).")

        try:
            if clear_before_import:
                traces.clear()
                tick_import_status("cleared existing traces")

            if suffix == ".npz":
                tick_import_status("reading NPZ")
                meta, imported_traces = read_npz_import(path)
                finalize_import(imported_traces, meta, "NPZ")
                return

            if suffix == ".mat":
                tick_import_status("reading MAT")
                try:
                    meta, imported_traces = read_mat_import(path)
                except Exception as e:
                    status_var.set(f"MAT import requires scipy ({e}). Install with: pip install scipy")
                    return
                finalize_import(imported_traces, meta, "MAT")
                return

            tick_import_status("reading CSV")
            import_probe_obj = probe_registry.get_probe_by_name(probe_var.get()) or probe_registry.get_default_probe()
            import_probe_snapshot = probe_to_snapshot(import_probe_obj)
            meta, imported_traces = read_csv_import(path, default_probe_snapshot=import_probe_snapshot)
            finalize_import(imported_traces, meta, "CSV")
        except Exception as e:
            status_var.set(f"Import failed: {e}")

    clear_selected_button = ttk.Button(
        trace_actions_frame, text="Clear Selected", command=clear_selected_trace, style="Secondary.TButton"
    )
    clear_selected_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
    edit_selected_button = ttk.Button(
        trace_actions_frame, text="Edit Selected", command=edit_selected_trace_popup, style="Secondary.TButton"
    )
    edit_selected_button.grid(row=0, column=1, sticky="ew", padx=4)
    clear_all_button = ttk.Button(
        trace_actions_frame, text="Clear All", command=clear_all_traces, style="Secondary.TButton"
    )
    clear_all_button.grid(row=0, column=2, sticky="ew", padx=(4, 0))

    add_section_header(actions_frame, "Actions")
    update_button = ttk.Button(actions_frame, text="Capture + Plot", command=update_plot, style="Primary.TButton")
    update_button.grid(row=1, column=0, sticky="ew", pady=(4, 0))

    import_button = ttk.Button(
        actions_frame,
        text="Import",
        command=import_data,
        style="Secondary.TButton",
    )
    import_button.grid(row=2, column=0, sticky="ew", pady=(4, 0))

    add_section_header(export_frame, "Export")
    dataset_name_label = ttk.Label(export_frame, text="Dataset Name", style="Ui.TLabel")
    dataset_name_label.grid(row=1, column=0, sticky="w")
    filename_var = tk.StringVar(value=next_available_stem(default_save_dir, "output"))
    filename_entry = tk.Entry(
        export_frame,
        textvariable=filename_var,
        font=ui_font,
        bg=colors["entry_bg"],
        fg=colors["text_primary"],
        insertbackground=colors["text_primary"],
        relief="solid",
        highlightthickness=1,
        highlightbackground=colors["entry_border"],
        highlightcolor=colors["accent"],
        bd=0,
    )
    filename_entry.grid(row=2, column=0, sticky="ew")
    export_destination_var = tk.StringVar(value="default")
    destination_frame = ttk.Frame(export_frame, style="CardInner.TFrame")
    destination_frame.grid(row=3, column=0, sticky="ew", pady=(4, 0))
    destination_frame.columnconfigure(0, weight=1)
    destination_frame.columnconfigure(1, weight=1)
    save_default_radio = ttk.Radiobutton(
        destination_frame,
        text="Save to Internal",
        value="default",
        variable=export_destination_var,
        style="Ui.TRadiobutton",
    )
    save_default_radio.grid(row=0, column=0, sticky="w")
    save_usb_radio = ttk.Radiobutton(
        destination_frame,
        text="Save to USB",
        value="usb",
        variable=export_destination_var,
        style="Ui.TRadiobutton",
    )
    save_usb_radio.grid(row=0, column=1, sticky="w")
    save_dir_label = ttk.Label(export_frame, text="", style="Hint.TLabel", width=72)
    save_dir_label.grid(row=5, column=0, sticky="w", pady=(4, 0))

    def format_save_path_for_label(path: Path, max_chars: int = 42) -> str:
        text = str(path)
        if len(text) <= max_chars:
            return text
        keep = max(8, max_chars - 3)
        return f"...{text[-keep:]}"

    def resolve_save_dir():
        if export_destination_var.get() == "usb":
            usb_dir = get_usb_save_dir()
            if usb_dir is None:
                return None, "USB drive not detected"
            return usb_dir, "USB"
        return default_save_dir, "Default"

    def refresh_save_destination(*_):
        requested_stem = normalize_stem(filename_var.get())
        target_dir, mode = resolve_save_dir()
        if target_dir is None:
            save_dir_label.configure(text="Saving to (USB): Not detected")
            return
        short_path = format_save_path_for_label(target_dir)
        save_dir_label.configure(text=f"Saving to ({mode}): {short_path}")
        filename_var.set(next_available_stem(target_dir, requested_stem))

    def prepare_export_target():
        save_dir, _ = resolve_save_dir()
        if save_dir is None:
            status_var.set("USB save selected, but no USB drive was detected.")
            return None, None, None
        requested_stem = normalize_stem(filename_var.get())
        chosen_stem = next_available_stem(save_dir, requested_stem)
        if len(traces) == 0:
            status_var.set("Error: No plotted traces. Capture first.")
            return None, None, None
        return save_dir, requested_stem, chosen_stem

    def finalize_single_export(save_dir: Path, requested_stem: str, chosen_stem: str, kind: str, path: Path):
        filename_var.set(next_available_stem(save_dir, requested_stem))
        if chosen_stem != requested_stem:
            status_var.set(f"Name existed; saved {kind}: {path}")
        else:
            status_var.set(f"Saved {kind}: {path}")
        refresh_save_destination()

    def export_png():
        save_dir, requested_stem, chosen_stem = prepare_export_target()
        if save_dir is None:
            return
        png_path = save_dir / f"{chosen_stem}.png"
        try:
            fig.savefig(png_path)
            finalize_single_export(save_dir, requested_stem, chosen_stem, "PNG", png_path)
        except Exception as e:
            status_var.set(f"Error saving PNG: {e}")

    def export_csv():
        save_dir, requested_stem, chosen_stem = prepare_export_target()
        if save_dir is None:
            return
        csv_path = save_dir / f"{chosen_stem}.csv"
        try:
            write_csv_export(
                csv_path=csv_path,
                traces=traces,
                standard=standard_var.get(),
                limit_curve=curve_var.get(),
                selected_probe=probe_var.get(),
                window_selection=window_var.get(),
            )
            finalize_single_export(save_dir, requested_stem, chosen_stem, "CSV", csv_path)
        except Exception as e:
            status_var.set(f"Error saving CSV: {e}")

    def export_npz():
        save_dir, requested_stem, chosen_stem = prepare_export_target()
        if save_dir is None:
            return
        npz_path = save_dir / f"{chosen_stem}.npz"
        try:
            np.savez_compressed(
                npz_path,
                **build_binary_payload(
                    traces=traces,
                    standard=standard_var.get(),
                    limit_curve=curve_var.get(),
                    selected_probe=probe_var.get(),
                    window_selection=window_var.get(),
                ),
            )
            finalize_single_export(save_dir, requested_stem, chosen_stem, "NPZ", npz_path)
        except Exception as e:
            status_var.set(f"Error saving NPZ: {e}")

    def export_mat():
        save_dir, requested_stem, chosen_stem = prepare_export_target()
        if save_dir is None:
            return
        mat_path = save_dir / f"{chosen_stem}.mat"
        try:
            from scipy.io import savemat  # type: ignore
        except Exception as e:
            status_var.set(f"MAT export requires scipy ({e}). Install with: pip install scipy")
            return
        try:
            savemat(
                mat_path,
                build_binary_payload(
                    traces=traces,
                    standard=standard_var.get(),
                    limit_curve=curve_var.get(),
                    selected_probe=probe_var.get(),
                    window_selection=window_var.get(),
                ),
                do_compression=True,
            )
            finalize_single_export(save_dir, requested_stem, chosen_stem, "MAT", mat_path)
        except Exception as e:
            status_var.set(f"Error saving MAT: {e}")

    export_buttons_frame = ttk.Frame(export_frame, style="CardInner.TFrame")
    export_buttons_frame.grid(row=4, column=0, sticky="ew", pady=(4, 0))
    for i in range(4):
        export_buttons_frame.columnconfigure(i, weight=1)
    export_png_button = ttk.Button(export_buttons_frame, text="Export PNG", command=export_png, style="Secondary.TButton")
    export_png_button.grid(row=0, column=0, sticky="ew", padx=(0, 3))
    export_csv_button = ttk.Button(export_buttons_frame, text="Export CSV", command=export_csv, style="Secondary.TButton")
    export_csv_button.grid(row=0, column=1, sticky="ew", padx=3)
    export_mat_button = ttk.Button(export_buttons_frame, text="Export MAT", command=export_mat, style="Secondary.TButton")
    export_mat_button.grid(row=0, column=2, sticky="ew", padx=3)
    export_npz_button = ttk.Button(export_buttons_frame, text="Export NPZ", command=export_npz, style="Secondary.TButton")
    export_npz_button.grid(row=0, column=3, sticky="ew", padx=(3, 0))

    status_var = tk.StringVar(value="Ready")

    refresh_probe_menu()
    refresh_standard_menu()
    export_destination_var.trace_add("write", refresh_save_destination)
    standard_var.trace_add("write", refresh_curve_menu)
    standard_var.trace_add("write", rerender_from_selection)
    probe_var.trace_add("write", on_probe_changed)
    window_var.trace_add("write", on_window_changed)
    curve_var.trace_add("write", refresh_probe_menu)
    curve_var.trace_add("write", refresh_trace_listbox)
    curve_var.trace_add("write", rerender_from_selection)
    refresh_curve_menu()
    rerender_from_selection()
    refresh_trace_listbox()
    refresh_save_destination()
    trace_listbox.bind("<Double-Button-1>", edit_selected_trace_popup)
    trace_listbox.bind("<Return>", edit_selected_trace_popup)

    status_bar = ttk.Label(
        root,
        textvariable=status_var,
        anchor="w",
        padding=(12, 8),
        style="Status.TLabel",
    )
    status_bar.grid(row=1, column=0, sticky="ew")

    def update_ui_scale(*_):
        # Scale from usable right-panel area so controls fit above the status bar.
        STATUS_CLEARANCE_PX = 14
        width = control_card.winfo_width()
        height = control_card.winfo_height()
        if width <= 1:
            width = content_frame.winfo_width()
        if height <= 1:
            height = content_frame.winfo_height()
        if width <= 1:
            width = root.winfo_width()
        if height <= 1:
            height = root.winfo_height() - max(status_bar.winfo_height(), 0)
        width = max(width, 1)
        height = max(height - STATUS_CLEARANCE_PX, 1)
        ideal_ui_size = min(24, max(8, min(int(width / 64), int(height / 36))))
        if height < 900:
            ideal_ui_size = min(ideal_ui_size, 13)
        if height < 820:
            ideal_ui_size = min(ideal_ui_size, 12)
        if height < 740:
            ideal_ui_size = min(ideal_ui_size, 11)
        if height < 680:
            ideal_ui_size = min(ideal_ui_size, 10)
        if height < 620:
            ideal_ui_size = min(ideal_ui_size, 9)

        def apply_scale(ui_size: int):
            dense = height < 840
            very_dense = height < 760
            ultra_dense = height < 700
            section_size = min(20, ui_size + 1)

            if ui_font.cget("size") != ui_size:
                ui_font.configure(size=ui_size)
                if ultra_dense:
                    pad_x, pad_y = 4, 2
                elif very_dense:
                    pad_x, pad_y = 6, 3
                elif dense:
                    pad_x, pad_y = 8, 4
                else:
                    pad_x = max(10, int(ui_size * 0.75))
                    pad_y = max(6, int(ui_size * 0.5))
                style.configure("Ui.TMenubutton", padding=(pad_x, pad_y))
                style.configure("Primary.TButton", padding=(pad_x, pad_y))
                style.configure("Secondary.TButton", padding=(pad_x, pad_y))
                standard_menu["menu"].configure(font=ui_font)
                probe_menu["menu"].configure(font=ui_font)
                curve_menu["menu"].configure(font=ui_font)
                window_menu["menu"].configure(font=ui_font)
            if section_font.cget("size") != section_size:
                section_font.configure(size=section_size)

            section_pad = (
                (4, 4, 4, 4)
                if ultra_dense
                else ((6, 6, 6, 6) if very_dense else ((8, 8, 8, 8) if dense else (10, 10, 10, 10)))
            )
            source_frame.configure(padding=section_pad)
            traces_frame.configure(padding=section_pad)
            actions_frame.configure(padding=section_pad)
            if ultra_dense:
                export_frame.configure(padding=(4, 4, 4, 2))
            elif very_dense:
                export_frame.configure(padding=(6, 6, 6, 3))
            elif dense:
                export_frame.configure(padding=(8, 8, 8, 4))
            else:
                export_frame.configure(padding=(10, 10, 10, 6))
            control_frame.configure(
                padding=(
                    (4, 4, 4, 4)
                    if ultra_dense
                    else ((6, 6, 6, 6) if very_dense else ((8, 8, 8, 8) if dense else (12, 12, 12, 12)))
                )
            )
            content_frame.configure(
                padding=(
                    (6, 4, 6, 8)
                    if ultra_dense
                    else ((8, 6, 8, 10) if very_dense else ((12, 8, 12, 10) if dense else (16, 12, 16, 12)))
                )
            )
            status_bar.configure(padding=(8, 5) if dense else (12, 8))

            control_row_pad = 1 if ultra_dense else (2 if very_dense else (4 if dense else 8))
            source_row_pad = 0 if ultra_dense else (1 if very_dense else (2 if dense else 4))
            for i in range(8):
                control_frame.rowconfigure(i, pad=control_row_pad)
            for i in range(13):
                source_frame.rowconfigure(i, pad=source_row_pad)

            compact_pady = (1, 0) if ultra_dense else ((2, 0) if dense else (4, 0))
            trace_actions_frame.grid_configure(pady=(1, 0) if very_dense else (4, 0))
            manage_probes_button.grid_configure(pady=compact_pady)
            manage_standards_button.grid_configure(pady=compact_pady)
            update_button.grid_configure(pady=compact_pady)
            import_button.grid_configure(pady=compact_pady)
            destination_frame.grid_configure(pady=(1, 0) if very_dense else ((2, 0) if dense else (3, 0)))
            export_buttons_frame.grid_configure(pady=(0, 0) if very_dense else ((1, 0) if dense else (2, 0)))
            save_dir_label.grid_configure(pady=(1, 0) if dense else (2, 0))

            if height < 620:
                trace_rows = 1
            elif ultra_dense:
                trace_rows = 2
            elif very_dense:
                trace_rows = 3
            else:
                trace_rows = 4
            trace_listbox.configure(height=trace_rows)

            if ultra_dense:
                manage_probes_button.configure(text="Probes")
                manage_standards_button.configure(text="Standards")
                update_button.configure(text="Capture")
                import_button.configure(text="Import")
                save_default_radio.configure(text="Default")
                save_usb_radio.configure(text="USB")
                export_png_button.configure(text="PNG")
                export_csv_button.configure(text="CSV")
                export_mat_button.configure(text="MAT")
                export_npz_button.configure(text="NPZ")
                clear_selected_button.configure(text="Clear Sel")
                edit_selected_button.configure(text="Edit Sel")
                clear_all_button.configure(text="Clear All")
                dataset_name_label.configure(text="Name")
            elif very_dense:
                manage_probes_button.configure(text="Manage Probes")
                manage_standards_button.configure(text="Manage Standards")
                update_button.configure(text="Capture + Plot")
                import_button.configure(text="Import")
                save_default_radio.configure(text="Save to Internal")
                save_usb_radio.configure(text="Save to USB")
                export_png_button.configure(text="Export PNG")
                export_csv_button.configure(text="Export CSV")
                export_mat_button.configure(text="Export MAT")
                export_npz_button.configure(text="Export NPZ")
                clear_selected_button.configure(text="Clear Selected")
                edit_selected_button.configure(text="Edit Selected")
                clear_all_button.configure(text="Clear All")
                dataset_name_label.configure(text="Dataset Name")
            else:
                manage_probes_button.configure(text="Manage Probes")
                manage_standards_button.configure(text="Manage Standards")
                update_button.configure(text="Capture + Plot")
                import_button.configure(text="Import")
                save_default_radio.configure(text="Save to Internal")
                save_usb_radio.configure(text="Save to USB")
                export_png_button.configure(text="Export PNG")
                export_csv_button.configure(text="Export CSV")
                export_mat_button.configure(text="Export MAT")
                export_npz_button.configure(text="Export NPZ")
                clear_selected_button.configure(text="Clear Selected")
                edit_selected_button.configure(text="Edit Selected")
                clear_all_button.configure(text="Clear All")
                dataset_name_label.configure(text="Dataset Name")

        min_ui_size = 7
        candidate_ui_size = ideal_ui_size
        while True:
            apply_scale(candidate_ui_size)
            root.update_idletasks()
            required_h = control_frame.winfo_reqheight()
            available_h = max(control_card.winfo_height() - STATUS_CLEARANCE_PX, 1)
            if required_h <= available_h or candidate_ui_size <= min_ui_size:
                break
            candidate_ui_size -= 1

        # Favor plot area width while keeping control labels readable.
        if width < 1400:
            content_frame.columnconfigure(0, weight=13)
            content_frame.columnconfigure(1, weight=7)
        elif height < 920:
            content_frame.columnconfigure(0, weight=7)
            content_frame.columnconfigure(1, weight=3)
        else:
            content_frame.columnconfigure(0, weight=7)
            content_frame.columnconfigure(1, weight=3)

        # Keep plot typography in sync with current UI scale after resize.
        selected_curve_for_scale, _ = get_selected_curve()
        render_plot(selected_curve_for_scale)

    def on_close():
        nonlocal scope
        if scope is not None:
            try:
                scope.close()
            except Exception:
                pass
        root.destroy()

    resize_job = None
    RESIZE_IDLE_MS = 220
    STARTUP_SCALE_DELAY_MS = 260
    ui_scale_ready = False
    last_root_size = (0, 0)

    def schedule_ui_scale(event=None):
        nonlocal resize_job, last_root_size
        if event is not None and event.widget is not root:
            return
        if not ui_scale_ready:
            return
        current_size = (max(root.winfo_width(), 1), max(root.winfo_height(), 1))
        if current_size == last_root_size:
            return
        last_root_size = current_size
        if resize_job is not None:
            root.after_cancel(resize_job)
        resize_job = root.after(RESIZE_IDLE_MS, run_ui_scale)

    def run_ui_scale():
        nonlocal resize_job, ui_scale_ready, last_root_size
        resize_job = None
        update_ui_scale()
        last_root_size = (max(root.winfo_width(), 1), max(root.winfo_height(), 1))
        ui_scale_ready = True

    root.bind("<Configure>", schedule_ui_scale)
    root.after(STARTUP_SCALE_DELAY_MS, run_ui_scale)
    root.protocol("WM_DELETE_WINDOW", on_close)

    root.mainloop()
