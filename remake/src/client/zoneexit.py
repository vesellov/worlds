"""
The exits of a zone which the party walks through (G16): an entrance has a #remove rectangle; when the
hero steps into it, the party goes to the zone and the entrance of the neighbour (LeaveToZone 66f2d0,
590700). The entrances which lead to a zone of the global map (a #zone line of map.txt without .mpr, for
example gz1g_gz2g) open the global map of the allod (client.globalmap).
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

from eilib import zones as zones_module

#------------------------------------------------------------------------------

# how often the remake says that a walk exit cannot be taken (milliseconds)
MESSAGE_DELAY_MS = 4000.0
# there is no zone of the name of the entrance
NO_ZONE_MESSAGE = 'The zone of the entrance is not found'

#------------------------------------------------------------------------------


class ZoneExit(object):
    """
    The walk exits of the current map.
    """

    def __init__(self, app):
        self.app = app
        self.map_name = None
        self.zone = None
        self.ticks_ms = 0.0
        self.wait_ms = 0.0
        self.last_exit = None

    def __repr__(self):
        return 'ZoneExit(%s)' % (self.zone.name if self.zone is not None else None)

    def update(self, dt_ms=55.0):
        app = self.app
        if app.unit is None or app.terrain is None:
            return
        if app.map_name != self.map_name:
            self.map_name = app.map_name
            self.zone = app.game_data.zone_of_map(app.map_name) if app.map_name else None
            self.last_exit = None
            self.wait_ms = 0.0
        if self.zone is None:
            return
        if self.wait_ms > 0.0:
            self.wait_ms = max(0.0, self.wait_ms - dt_ms)
        zone_exit = zones_module.remove_exit(self.zone, app.unit.x, app.unit.y)
        if zone_exit is None:
            self.last_exit = None
            return
        if zone_exit is self.last_exit:
            return
        self.last_exit = zone_exit
        self.go(zone_exit)

    def go(self, zone_exit):
        """
        The party leaves the zone through the entrance: the neighbour zone with an .mpr is entered by its
        entrance; a zone of the global map is reported.
        """
        app = self.app
        zone = None
        for candidate in app.game_data.zones():
            if candidate.name.lower() == (zone_exit.neighbour_name or '').lower():
                zone = candidate
                break
        if zone is None:
            if self.wait_ms <= 0.0:
                app.show_message(NO_ZONE_MESSAGE)
                self.wait_ms = MESSAGE_DELAY_MS
            return
        if not zone.mpr_name:
            app.open_global_map(zone, zone_exit.pass_time)
            return
        app.advance_hours(zone_exit.pass_time)
        app.leave_to_zone(zone.name, zone_exit.neighbour_exit)
