import tkinter as tk
from tkinter import ttk
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import ttkbootstrap as tb
import tkintermapview
import math

class SunAnimation(tk.Canvas):
    def __init__(self, parent, width=800, height=200):
        super().__init__(parent, width=width, height=height, bg='#1B2838')
        self.width = width
        self.height = height
        
        # Center point for the arc
        self.center_x = width // 2
        self.center_y = height - 20  # Bottom padding
        
        # Arc radius
        self.radius = min(width // 3, height - 40)
        
        # Building dimensions (make it more symbolic)
        self.building_width = 60
        self.building_height = 50
        
        # Draw static elements
        self.draw_static_elements()
        self.sun = None
        
    def draw_static_elements(self):
        # Draw time labels and tick marks along the arc
        times = ['6:00', '9:00', '12:00', '15:00', '18:00']
        for i, time in enumerate(times):
            # Calculate position along the arc
            angle = math.pi * (0.2 + i * 0.15)  # Spread across the arc
            x = self.center_x + self.radius * math.cos(angle)
            y = self.center_y - self.radius * math.sin(angle)
            
            # Draw tick mark
            tick_length = 10
            tick_end_x = self.center_x + (self.radius - tick_length) * math.cos(angle)
            tick_end_y = self.center_y - (self.radius - tick_length) * math.sin(angle)
            self.create_line(x, y, tick_end_x, tick_end_y, fill='#4B6075', width=2)
            
            # Draw time label
            label_x = self.center_x + (self.radius + 20) * math.cos(angle)
            label_y = self.center_y - (self.radius + 20) * math.sin(angle)
            self.create_text(label_x, label_y, text=time, fill='#8B9BA9', 
                           font=('Arial', 8, 'bold'))
        
        # Draw reference arc
        arc_bbox = (
            self.center_x - self.radius,
            self.center_y - self.radius,
            self.center_x + self.radius,
            self.center_y + self.radius
        )
        self.create_arc(arc_bbox, start=0, extent=180, 
                       style='arc', outline='#4B6075', width=2)
        
        # Draw modern minimalistic house
        house_width = 40
        house_height = 35
        house_x = self.center_x - house_width // 2
        house_y = self.center_y - 5  # Slight offset from bottom
        
        # House base (simple rectangle)
        self.create_rectangle(
            house_x,
            house_y - house_height,
            house_x + house_width,
            house_y,
            fill='#2A2A2A',
            outline='#45B7D1',
            width=1
        )
        
        # Flat roof with solar panels
        panel_margin = 2
        panel_height = 8
        panel_segments = 3
        panel_width = (house_width - (panel_segments + 1) * panel_margin) / panel_segments
        
        for i in range(panel_segments):
            x = house_x + panel_margin + i * (panel_width + panel_margin)
            y = house_y - house_height - panel_height
            self.create_rectangle(
                x, y,
                x + panel_width, y + panel_height,
                fill='#45B7D1',
                outline='#45B7D1',
                width=1
            )
        
    def update_sun_position(self, timestamp, irradiation):
        if self.sun:
            self.delete(self.sun)
            
        # Parse hour and minute from timestamp
        dt = datetime.strptime(timestamp, '%d.%m.%Y %H:%M')
        hour = dt.hour + dt.minute / 60
        
        # Calculate sun position (only show sun between 6:00 and 18:00)
        if 6 <= hour <= 18:
            # Convert hour to angle (6:00 = 180°, 18:00 = 0°)
            angle = math.pi * (1 - (hour - 6) / 12)
            
            # Calculate position
            x = self.center_x + self.radius * math.cos(angle)
            y = self.center_y - self.radius * math.sin(angle)
            
            # Draw sun with size based on irradiation
            sun_radius = max(10, min(15, 10 + irradiation / 200))
            self.sun = self.create_oval(
                x - sun_radius, y - sun_radius,
                x + sun_radius, y + sun_radius,
                fill='#FFD700',
                outline='#FFA500',
                width=2
            )
            
            # Add rays if there's irradiation
            if irradiation > 0:
                self.draw_sun_rays(x, y, sun_radius, irradiation)
    
    def draw_sun_rays(self, x, y, radius, irradiation):
        ray_count = 8
        ray_length = radius + min(15, irradiation / 100)
        for i in range(ray_count):
            angle = 2 * math.pi * i / ray_count
            x1 = x + radius * math.cos(angle)
            y1 = y + radius * math.sin(angle)
            x2 = x + ray_length * math.cos(angle)
            y2 = y + ray_length * math.sin(angle)
            self.create_line(x1, y1, x2, y2, fill='#FFD700', width=2)

class ForecastUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PV Power Forecast")
        
        # Store the style object as instance variable
        # Configure custom style
        self.style = tb.Style(theme='darkly')
        
        # Initialize data loading
        self.root.after(100, self.show_forecast)  # Load data after UI is initialized
        
        # Configure custom colors and styles
        self.style.configure('Custom.TFrame', background='#1A1A1A')
        self.style.configure('Card.TFrame',
                            background='#2A2A2A',
                            borderwidth=1,
                            relief='solid',
                            padding=15)
        self.style.configure('Stats.TLabel',
                            font=('Segoe UI', 10),
                            foreground='#888888',
                            background='#2A2A2A')
        self.style.configure('Value.TLabel',
                            font=('Segoe UI', 32, 'bold'),
                            foreground='#45B7D1',
                            background='#2A2A2A')
        
        # Initialize status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(
            root,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            padding=(10, 2)
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Configure notebook style
        self.style.configure('TNotebook',
                           background='#1A1A1A',
                           borderwidth=0)
        self.style.configure('TNotebook.Tab',
                           background='#2A2A2A',
                           foreground='#888888',
                           padding=[20, 5],
                           font=('Segoe UI', 9))
        self.style.map('TNotebook.Tab',
                      background=[('selected', '#1A1A1A')],
                      foreground=[('selected', '#FFFFFF')])
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Create tabs
        self.forecast_tab = ttk.Frame(self.notebook)
        self.graph_tab = ttk.Frame(self.notebook)
        self.stats_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.forecast_tab, text='☀ Forecast')
        self.notebook.add(self.graph_tab, text=' Graph')
        self.notebook.add(self.stats_tab, text='📊 Location')
        
        # Initialize the map in stats tab
        self.create_map_view()
        
        # Create main frame in forecast tab with dark background
        self.main_frame = ttk.Frame(self.forecast_tab, style='Custom.TFrame', padding="20")
        self.main_frame.pack(fill='both', expand=True)
        
        # Create stats cards frame
        self.stats_frame = ttk.Frame(self.main_frame, style='Custom.TFrame')
        self.stats_frame.pack(fill='x', pady=(0, 20))
        
        # Create three card frames for key stats
        for i, (title, value, unit) in enumerate([
            ("Current Power", "0", "W"),
            ("Current Temperature", "0", "°C"),
            ("Solar Irradiance", "0", "W/m²")
        ]):
            # Create card with fixed minimum width
            card = ttk.Frame(self.stats_frame, style='Card.TFrame', width=300)
            card.grid(row=0, column=i, padx=10, sticky='ew')
            card.grid_propagate(False)  # Maintain fixed size
            self.stats_frame.columnconfigure(i, weight=1)
            
            # Title at top
            title_label = ttk.Label(card, text=title, style='Stats.TLabel')
            title_label.pack(anchor='w', pady=(0,5))
            
            # Center frame for value and unit
            center_frame = ttk.Frame(card, style='Card.TFrame')
            center_frame.pack(expand=True, fill='both')
            
            # Value and unit container
            value_container = ttk.Frame(center_frame, style='Card.TFrame')
            value_container.pack(expand=True, anchor='center')
            
            # Value and unit on same line
            ttk.Label(value_container, text=value,
                     style='Value.TLabel',
                     font=('Segoe UI', 28, 'bold')).pack(side='left')
            ttk.Label(value_container,
                     text=f" {unit}",
                     style='Value.TLabel',
                     font=('Segoe UI', 14, 'bold'),
                     foreground='#888888').pack(side='left', padx=(2,0))
        
        # Create Treeview in a card frame
        self.data_card = ttk.Frame(self.main_frame, style='Card.TFrame')
        self.data_card.pack(fill='both', expand=True)
        
        # Add title for the data section
        ttk.Label(self.data_card,
                 text="Forecast Data",
                 style='Stats.TLabel').pack(anchor='w', pady=(0, 10))
        
        self.create_treeview()
        
    def update_stats_cards(self, power, temperature, irradiance):
        """Update the stats cards with current values"""
        # Find all Value.TLabel widgets in the stats frame
        for card in self.stats_frame.winfo_children():
            value_container = card.winfo_children()[1].winfo_children()[0]
            value_label = value_container.winfo_children()[0]
            
            # Update value based on the card's position
            if "Current Power" in card.winfo_children()[0].cget("text"):
                value_label.configure(text=f"{power:.1f}")
            elif "Current Temperature" in card.winfo_children()[0].cget("text"):
                value_label.configure(text=f"{temperature:.1f}")
            elif "Solar Irradiance" in card.winfo_children()[0].cget("text"):
                value_label.configure(text=f"{irradiance:.1f}")
        
        # Configure refresh button style
        self.style.configure('Refresh.TButton',
                           background='#45B7D1',
                           foreground='white',
                           padding=[15, 8],
                           font=('Segoe UI', 9))
        
        # Create button container for centering
        button_container = ttk.Frame(self.data_card, style='Card.TFrame')
        button_container.pack(fill='x', pady=(10, 0))
        
        # Create and style the refresh button
        self.forecast_button = ttk.Button(
            button_container,
            text="↻ Refresh Forecast",
            command=self.show_forecast,
            style='Refresh.TButton'
        )
        self.forecast_button.pack(anchor='center', pady=10)

    def show_forecast(self):
        self.status_var.set("Loading forecast data...")
        self.root.update()
        
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        try:
            df = pd.read_csv('data/forecast_data.csv')
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Get current values for stats cards
            current_time = datetime.now()
            current_data = df.iloc[0]  # Get first row for current values
            
            # Update stats cards with current values
            self.update_stats_cards(
                power=current_data['power_w'],
                temperature=current_data['temperature_c'],
                irradiance=current_data['global_irradiation']
            )
            
            # Get forecast data (show all available data)
            forecast_data = df[df['timestamp'] > current_time]
            
            # Update treeview with forecast data
            for _, row in forecast_data.iterrows():
                values = (
                    row['timestamp'].strftime('%d.%m.%Y %H:%M'),
                    f"{row['power_w']:.1f}",
                    f"{row['temperature_c']:.1f}",
                    f"{row['wind_speed_ms']:.1f}",
                    f"{row['global_irradiation']:.1f}"
                )
                self.tree.insert("", "end", values=values)
            
            self.update_graphs(forecast_data)
            self.status_var.set("Forecast updated successfully")
            
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            print(f"Error loading forecast data: {e}")

    def create_treeview(self):
        """Create and configure the Treeview widget"""
        self.forecast_frame = ttk.Frame(self.data_card)
        self.forecast_frame.pack(fill='both', expand=True)
        
        # Configure Treeview style
        style = ttk.Style()
        style.configure("Treeview",
                       background="#2A2A2A",
                       foreground="#CCCCCC",
                       fieldbackground="#2A2A2A",
                       rowheight=35,
                       font=('Segoe UI', 9))
        
        style.configure("Treeview.Heading",
                       background="#1A1A1A",
                       foreground="#888888",
                       relief="flat",
                       font=('Segoe UI', 9))
        
        style.map("Treeview",
                 background=[('selected', '#333333')],
                 foreground=[('selected', '#45B7D1')])
        
        self.tree = ttk.Treeview(
            self.forecast_frame,
            columns=("Time", "Power", "Temperature", "Wind", "Irradiation"),
            show="headings",
            height=12
        )
        
        columns = {
            "Time": ("Time", 150),
            "Power": ("AC Power (W)", 120),
            "Temperature": ("Temperature (°C)", 120),
            "Wind": ("Wind Speed (m/s)", 120),
            "Irradiation": ("Irradiation (W/m²)", 120)
        }
        
        for col, (text, width) in columns.items():
            self.tree.heading(col, text=text)
            self.tree.column(col, width=width, anchor="center")
        
        style = ttk.Style()
        style.configure("Treeview", rowheight=30)
        
        scrollbar = ttk.Scrollbar(self.forecast_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        
        self.forecast_frame.columnconfigure(0, weight=1)
        
        # Add sun animation below treeview
        self.sun_animation = SunAnimation(self.forecast_frame)
        self.sun_animation.grid(row=1, column=0, columnspan=2, sticky="ew", pady=10)
        
        # Bind treeview selection to update sun position
        self.tree.bind('<<TreeviewSelect>>', self.update_sun_animation)
        
    def update_sun_animation(self, event):
        selection = self.tree.selection()
        if selection:
            item = self.tree.item(selection[0])
            values = item['values']
            timestamp = values[0]  # Time column
            irradiation = float(values[4])  # Irradiation column
            self.sun_animation.update_sun_position(timestamp, irradiation)

    def update_graphs(self, forecast_data):
        # Clear previous widgets in graph tab
        for widget in self.graph_tab.winfo_children():
            widget.destroy()
            
        # Create figure with subplots and dark background
        plt.style.use('default')
        fig = plt.figure(figsize=(12, 8))
        fig.patch.set_facecolor('#1A1A1A')  # Match UI background
        
        # Power plot
        ax1 = plt.subplot(2, 1, 1)
        ax1.set_facecolor('#2A2A2A')  # Match card background
        ax1.plot(forecast_data['timestamp'], 
                forecast_data['power_w'], 
                color='#45B7D1',  # Bright blue
                linewidth=3,
                label='AC Power',
                marker='o',
                markersize=8,
                markerfacecolor='white')
        
        # Customize first plot
        ax1.set_ylabel('Power (W)', fontsize=12, fontweight='bold', color='white')
        ax1.set_title('Power Forecast', fontsize=14, fontweight='bold', color='white', pad=20)
        ax1.grid(True, linestyle='--', alpha=0.2, color='white')
        ax1.legend(facecolor='#2B3E50', edgecolor='none', fontsize=10, labelcolor='white')
        ax1.tick_params(colors='white')
        ax1.spines['bottom'].set_color('white')
        ax1.spines['top'].set_color('white')
        ax1.spines['left'].set_color('white')
        ax1.spines['right'].set_color('white')

        # Temperature and Wind Speed plot
        ax2 = plt.subplot(2, 1, 2)
        ax2.set_facecolor('#2A2A2A')  # Match card background
        
        # Plot temperature
        line1 = ax2.plot(forecast_data['timestamp'], 
                        forecast_data['temperature_c'], 
                        color='#FF7676',  # Soft red
                        linewidth=3,
                        label='Temperature',
                        marker='s',
                        markersize=8,
                        markerfacecolor='white')
        
        ax2.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold', color='white')
        ax2.grid(True, linestyle='--', alpha=0.2, color='white')
        ax2.tick_params(colors='white')

        # Add Wind Speed on secondary y-axis
        ax2_twin = ax2.twinx()
        line2 = ax2_twin.plot(forecast_data['timestamp'], 
                             forecast_data['wind_speed_ms'], 
                             color='#98D8AA',  # Soft green
                             linewidth=3,
                             label='Wind Speed',
                             marker='^',
                             markersize=8,
                             markerfacecolor='white')
        
        ax2_twin.set_ylabel('Wind Speed (m/s)', fontsize=12, fontweight='bold', color='white')
        ax2_twin.tick_params(colors='white')

        # Customize spines for both axes
        for ax in [ax2, ax2_twin]:
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['right'].set_color('white')

        # Combine legends with custom colors
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, 
                  loc='upper right', 
                  facecolor='#2B3E50', 
                  edgecolor='none', 
                  fontsize=10, 
                  labelcolor='white')

        # Customize x-axis for both plots
        for ax in [ax1, ax2]:
            ax.tick_params(axis='x', rotation=45)
            ax.set_xlabel('Time', fontsize=12, fontweight='bold', color='white')

        # Adjust layout
        plt.tight_layout()

        # Add plot to graph tab with padding
        canvas = FigureCanvasTkAgg(fig, master=self.graph_tab)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill='both', expand=True, padx=20, pady=20)

    def create_map_view(self):
        """Create and configure the map view"""
        # Create frame for map
        self.map_frame = ttk.Frame(self.stats_tab)
        self.map_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create map widget
        self.map_widget = tkintermapview.TkinterMapView(
            self.map_frame, 
            width=800, 
            height=600, 
            corner_radius=0
        )
        self.map_widget.pack(fill='both', expand=True)
        
        # Set location coordinates
        lat, lon = map(float, "47.38770748541585,15.094127778561258".split(','))
        
        # Set position and zoom
        self.map_widget.set_position(lat, lon)
        self.map_widget.set_zoom(15)
        
        # Add marker
        self.map_widget.set_marker(
            lat, lon, 
            text="PV Installation",
            marker_color_circle="blue",
            marker_color_outside="gray"
        )

def main():
    root = tb.Window(themename="darkly")
    
    try:
        # Load the pv.ico file
        root.iconbitmap('pv.ico')
        
        # Configure window appearance
        root.configure(background='#1A1A1A')
        root.title("⚡ PV Power Forecast")
        
        # Set window size and position
        window_width = 1000
        window_height = 800
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        center_x = int(screen_width/2 - window_width/2)
        center_y = int(screen_height/2 - window_height/2)
        root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        
        # Configure title bar and theme
        style = ttk.Style()
        style.configure('TitleBar.TFrame', background='#1A1A1A')
        
    except Exception as e:
        print(f"Could not load icon: {e}")
    
    app = ForecastUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
